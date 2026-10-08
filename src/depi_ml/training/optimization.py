"""Una ronda fija de optimización exploratoria, con referencia persistida y prueba vedada."""

import csv
import json
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from depi_ml.analysis.audit import audit_dataset
from depi_ml.analysis.dataset import Phase2Error, digest, load_dataset, new_output, write_json
from depi_ml.analysis.reports import plotting
from depi_ml.datasets.label_availability import attach_labels, load_label_availability
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.evaluation.metrics import binary_metrics
from depi_ml.training.experiments import EXPERIMENTS, json_parameters, source_hash, versions
from depi_ml.training.preprocessing import make_pipeline
from depi_ml.training.review import EXPLORATORY_WARNING, exploratory_review
from depi_ml.training.splits import temporal_split, utc_timestamp


TARGET_EXPERIMENTS = ("full", "without_catalog")
# Se fija antes de ejecutar; no hay segunda ronda, early stopping ni ajuste de umbral.
CONFIGURATIONS = {
    "random_forest": {
        "rf_leaf20": {"min_samples_leaf": 20},
        "rf_features03": {"max_features": .3},
        "rf_regularized": {"min_samples_leaf": 20, "max_features": .3, "max_depth": 16},
    },
    "xgboost": {
        "xgb_shallow": {"n_estimators": 300, "max_depth": 3, "min_child_weight": 5, "reg_lambda": 5},
        "xgb_depth5": {"n_estimators": 300, "max_depth": 5, "min_child_weight": 5, "reg_lambda": 5},
        "xgb_deep_regularized": {"n_estimators": 400, "max_depth": 6, "learning_rate": .03,
                                 "min_child_weight": 10, "reg_lambda": 10, "gamma": .1},
        "xgb_slow": {"n_estimators": 400, "learning_rate": .03, "min_child_weight": 5, "reg_lambda": 5},
        "xgb_unweighted": {"n_estimators": 300, "min_child_weight": 5, "reg_lambda": 5, "scale_pos_weight": 1},
    },
}
METRICS = ("roc_auc", "pr_auc_average_precision", "recall", "precision", "f1", "brier_score")
RELEVANCE = {"min_absolute_ap_gain": .01, "max_auc_decline": .002,
             "max_brier_increase": .002, "max_monthly_ap_decline": .01,
             "interpretation": "Descriptive predeclared screen, not statistical significance or operational validation"}
COMMERCIAL_CONTEXT = {
    "source": "User-provided operational context and prior audit; not recomputed using reserved test",
    "transition_months_approximate": ["2026-06", "2026-07"], "exact_effective_date": None,
    "before": "Reservations could be made before payment; payment on arrival was possible.",
    "after": "General rule: purchase a voucher before booking.",
    "exception": "Walk-ins may be attended the same day if capacity is available.",
    "provided_audit_no_show_rates": {"2026-06": .2092, "2026-07": .0715, "2026-08": .0478, "2026-09": .0258},
    "limitations": [
        "Possible change in booking behavior/distribution (concept drift); exclusive causality is not established.",
        "June-July 2026 is an approximate transition; no exact implementation day is invented.",
        "April-June validation may partly overlap transition; it does not measure stable post-change performance.",
        "Audit percentages supplied by the user may have different date bases, population and label maturity than purged validation.",
        "Eligibility, clinic/customer mix, tracking, label maturity and walk-in selection may also affect observed rates.",
        "Policy context is not a new predictor; cuts, targets, original features and candidate plan remain unchanged.",
    ],
}


def check_plan():
    if set(CONFIGURATIONS) != {"random_forest", "xgboost"}:
        raise Phase2Error("La ronda requiere Random Forest y XGBoost.")
    for candidates in CONFIGURATIONS.values():
        if not candidates or len(candidates) + 1 > 8 or "original" in candidates:
            raise Phase2Error("Máximo de 8 configuraciones por algoritmo/experimento, incluida la original.")
        if any(set(params) & {"random_state", "n_jobs", "early_stopping_rounds", "callbacks"}
               for params in candidates.values()):
            raise Phase2Error("No cambiar semilla, hilos ni introducir rondas adaptativas/early stopping.")


def monthly_metrics(frame, probability):
    """Mes local de reserva; solo filas ya retenidas de validación, sin IDs exportados."""
    probability = np.asarray(probability, dtype=float)
    month = frame.prediction_at.dt.tz_convert("America/Lima").dt.strftime("%Y-%m").to_numpy()
    y = frame.target.to_numpy(dtype=int)
    return [{"month": value, "date_basis": "prediction_at America/Lima", "status": EXPLORATORY_WARNING,
             "evaluation_partition": "validation", **binary_metrics(y[month == value], probability[month == value], .5)}
            for value in sorted(set(month))]


def compare_to_original(metrics, monthly, original, original_monthly):
    delta = {key: metrics[key] - original[key] for key in METRICS}
    reference_months = {row["month"]: row for row in original_monthly}
    monthly_delta = {row["month"]: (row["pr_auc_average_precision"] - reference_months[row["month"]]["pr_auc_average_precision"])
                     if row["pr_auc_average_precision"] is not None and reference_months[row["month"]]["pr_auc_average_precision"] is not None
                     else None for row in monthly}
    evaluable = all(value is not None for value in monthly_delta.values())
    relevant = (delta["pr_auc_average_precision"] >= RELEVANCE["min_absolute_ap_gain"] and
                delta["roc_auc"] >= -RELEVANCE["max_auc_decline"] and
                delta["brier_score"] <= RELEVANCE["max_brier_increase"] and evaluable and
                all(value >= -RELEVANCE["max_monthly_ap_decline"] for value in monthly_delta.values()))
    return {"delta_vs_original": delta, "monthly_ap_delta_vs_original": monthly_delta,
            "descriptive_relevance": "passes_predeclared_screen" if relevant else "no_clear_relevant_improvement",
            "selection_status": "no_definitive_model_selected"}


def read_reference(root, dataset, labels):
    path = root / "experiment.json"
    try:
        sha = digest(path)
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Phase2Error("Referencia local incompleta o inválida.") from None
    if (not isinstance(metadata, dict) or metadata.get("artifact_format_version") != 2 or
            metadata.get("status") != EXPLORATORY_WARNING or metadata.get("training_mode") != "exploratory"):
        raise Phase2Error("La referencia debe ser el experimento exploratorio original terminado.")
    if (metadata.get("dataset_sha256") != dataset.sha256 or metadata.get("label_availability") != labels.identity() or
            metadata.get("predictor_contract") != PREDICTOR_COLUMNS or metadata.get("versions") != versions()):
        raise Phase2Error("Hashes, auxiliar, contrato o versiones distintos de la referencia original.")
    if metadata.get("seed") != 42 or metadata.get("threshold") != .5:
        raise Phase2Error("La referencia debe conservar semilla 42 y umbral 0.5.")
    expected = {(name, model) for name in TARGET_EXPERIMENTS for model in ["baseline", *CONFIGURATIONS]}
    try:
        entries = [entry for entry in metadata["models"] if entry["experiment"] in TARGET_EXPERIMENTS]
        if len(entries) != len(expected) or {(e["experiment"], e["model"]) for e in entries} != expected:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise Phase2Error("Faltan modelos originales/baselines o hay entradas duplicadas en la referencia.") from None
    hashes = {path: sha}
    for entry in entries:
        model = (root / entry["model_path"]).resolve()
        if root not in model.parents or not model.is_file() or digest(model) != entry["model_sha256"]:
            raise Phase2Error("Ruta o hash de modelo original inválido; no se deserializa.")
        if entry["features"] != EXPERIMENTS[entry["experiment"]]:
            raise Phase2Error("Predictoras originales distintas del diseño requerido.")
        if (entry["split"].get("test_partition_materialized") is not False or
                entry["validation_metrics"].get("status") != EXPLORATORY_WARNING):
            raise Phase2Error("Referencia sin aislamiento exploratorio de prueba.")
        if entry["hyperparameters"].get("random_state") != 42:
            raise Phase2Error("Semilla del modelo original distinta de 42.")
        hashes[model] = entry["model_sha256"]
    return metadata, entries, hashes


def save_tables(out, entries):
    comparison = [{"experiment": e["experiment"], "model": e["model"], "configuration": e["configuration"],
                   "role": e["role"], **e["validation_metrics"], **e.get("comparison", {})} for e in entries]
    monthly = [{"experiment": e["experiment"], "model": e["model"], "configuration": e["configuration"],
                "role": e["role"], **row} for e in entries for row in e["monthly_metrics"]]
    write_json(out / "validation_comparison.json", comparison)
    write_json(out / "validation_monthly.json", monthly)
    for filename, rows in [("validation_comparison.csv", comparison), ("validation_monthly.csv", monthly)]:
        columns = ["experiment", "model", "configuration", "role"]
        if filename == "validation_monthly.csv":
            columns += ["month", "date_basis"]
        columns += ["rows", "no_show", "prevalence", *METRICS, "threshold", "status", "evaluation_partition"]
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    lines = [EXPLORATORY_WARNING, "", "Una ronda fija; semilla 42; umbral diagnóstico 0.5. Solo entrenamiento/validación.",
             "Originales reevaluados sin reentrenar. Prueba no materializada ni evaluada.", "",
             "| Experimento | Modelo | Configuración | ROC-AUC | AP | Recall | Precision | F1 | Brier | ΔAP original |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison:
        values = [f"{row[key]:.4f}" for key in METRICS]
        delta = row.get("delta_vs_original", {}).get("pr_auc_average_precision")
        lines.append("| " + " | ".join([row["experiment"], row["model"], row["configuration"], *values,
                                         f"{delta:+.4f}" if delta is not None else "—"]) + " |")
    lines += ["", "Relevancia descriptiva predeclarada: ΔAP >=0.01, caída ROC-AUC <=0.002, aumento Brier <=0.002 "
              "y caída AP por mes <=0.01. No representa significancia estadística ni validación operativa.", "",
              "| Experimento | Modelo | Configuración | Mes de reserva | N | Prevalencia | ROC-AUC | AP | Recall | Precision | F1 | Brier |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in monthly:
        values = [f"{row[key]:.4f}" if row[key] is not None else "N/A" for key in METRICS]
        lines.append("| " + " | ".join([row["experiment"], row["model"], row["configuration"], row["month"],
                                        str(row["rows"]), f"{row['prevalence']:.4f}", *values]) + " |")
    lines += ["", "Cambio comercial y limitaciones:", "",
              "DEPI pasó de permitir reservas con pago al llegar a exigir generalmente compra de bono antes de reservar. "
              "La transición se sitúa aproximadamente entre junio y julio de 2026; no se conoce el día exacto. "
              "Se permite atención presencial el mismo día si hay disponibilidad.", "",
              "La auditoría previa aportada por el usuario indicó tasas de candidatos no-show: junio 20.92%, julio 7.15%, "
              "agosto 4.78%, septiembre 2.58%. Son antecedentes externos a esta ronda: no se recalcularon "
              "los períodos de prueba ni se midió desempeño predictivo en ellos.", "",
              *[f"- {value}" for value in COMMERCIAL_CONTEXT["limitations"]],
              "- Reutilizar validación para comparar hiperparámetros introduce optimismo de selección; no hay confirmación independiente ni intervalos de incertidumbre.",
              "- Los ocho controles metodológicos pueden continuar pendientes. No se selecciona ni promueve un modelo definitivo."]
    (out / "optimization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_monthly(out, entries):
    plt = plotting()
    for experiment in TARGET_EXPERIMENTS:
        for model in CONFIGURATIONS:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for entry in entries:
                if entry["experiment"] != experiment or entry["model"] not in [model, "baseline"]:
                    continue
                rows = entry["monthly_metrics"]
                for axis, metric in zip(axes, ["pr_auc_average_precision", "brier_score"]):
                    axis.plot([r["month"] for r in rows], [r[metric] for r in rows], marker="o",
                              label=entry["configuration"] if entry["model"] == model else "baseline",
                              linewidth=2 if entry["configuration"] == "original" else 1)
                    axis.set(xlabel="Mes de reserva (Lima)", ylabel=metric)
            axes[1].legend(fontsize=7)
            fig.suptitle(f"{EXPLORATORY_WARNING}: {experiment} / {model}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"monthly_{experiment}_{model}.png")
            plt.close(fig)


def optimize(csv_path, manifest_path, labels_path, labels_manifest_path, reference_path, output, review_path=None):
    check_plan()
    code_sha = source_hash()
    dataset = load_dataset(csv_path, manifest_path)
    labels = load_label_availability(dataset, labels_path, labels_manifest_path)
    frame = attach_labels(dataset, labels)
    root = Path(reference_path).resolve()
    reference, original_entries, protected = read_reference(root, dataset, labels)
    requested_out = Path(output).resolve()
    if root == requested_out or root in requested_out.parents or requested_out in root.parents:
        raise Phase2Error("La salida debe ser nueva y separada de los resultados originales.")
    splits = {}
    # Comprobar todos los cortes originales antes de deserializar/ajustar modelos.
    for entry in original_entries:
        recorded = entry["split"]
        split = temporal_split(frame, recorded["validation_start"], recorded["test_start"], recorded["test_end"],
                               dataset.manifest["exported_at"], include_test=False)
        if split.metadata != recorded:
            raise Phase2Error("Cortes/particiones recalculados distintos de los originales.")
        splits[entry["experiment"]] = split
    test_start = original_entries[0]["split"]["test_start"]
    if any(entry["split"]["test_start"] != test_start for entry in original_entries):
        raise Phase2Error("Los experimentos originales no comparten el corte de prueba.")
    audit = audit_dataset(dataset, descriptive_before=utc_timestamp(test_start))
    review = exploratory_review(dataset, audit, review_path, labels)
    audit.update(training_status=EXPLORATORY_WARNING, pending_review=review["pending_controls"])
    protected.update({dataset.source: dataset.sha256, dataset.manifest_source: dataset.manifest_sha256,
                      labels.source: labels.sha256,
                      Path(labels_manifest_path).resolve() if labels_manifest_path is not None else
                      labels.source.with_name(labels.source.stem + "_manifest.json"): labels.manifest_sha256})
    if review_path is not None:
        protected[Path(review_path).resolve()] = review["original_review_sha256"]
    # Validar predicciones/contrato originales contra las métricas conservadas. No refit.
    references = {}
    for entry in original_entries:
        pipeline = joblib.load((root / entry["model_path"]).resolve())
        if (pipeline.named_steps["contract"].features_ != entry["features"] or
                json_parameters(pipeline.named_steps["model"].get_params()) != entry["hyperparameters"]):
            raise Phase2Error("Contrato/hiperparámetros persistidos distintos de la referencia.")
        validation = splits[entry["experiment"]].validation
        probability = pipeline.predict_proba(validation[entry["features"]])[:, 1]
        metrics = binary_metrics(validation.target, probability, .5)
        if any(not np.isclose(metrics[key], entry["validation_metrics"][key], atol=1e-12, rtol=1e-12)
               for key in METRICS) or metrics["confusion_matrix"] != entry["validation_metrics"]["confusion_matrix"]:
            raise Phase2Error("Predicciones originales no reproducen las métricas de validación conservadas.")
        references[(entry["experiment"], entry["model"])] = (metrics, monthly_metrics(validation, probability))
        del pipeline
    out = new_output(requested_out, dataset)
    plan = {"status": EXPLORATORY_WARNING, "rounds": 1, "seed": 42, "threshold": .5,
            "experiments": list(TARGET_EXPERIMENTS), "candidate_overrides": CONFIGURATIONS,
            "configurations_per_algorithm_experiment_including_original": {name: len(configs) + 1 for name, configs in CONFIGURATIONS.items()},
            "relevance_policy": RELEVANCE, "commercial_context": COMMERCIAL_CONTEXT,
            "selection_policy": "Fixed plan before fits; validation ranking descriptive; no second round or definitive selection"}
    write_json(out / "plan.json", plan)
    plan_sha = digest(out / "plan.json")
    write_json(out / "audit.json", audit)
    write_json(out / "review.json", review)
    entries = []
    marker = {"status": EXPLORATORY_WARNING, "evaluation_partition": "validation"}
    for original in original_entries:
        metrics, monthly = references[(original["experiment"], original["model"])]
        entries.append({**original, "configuration": "original", "role": "reference_not_refit",
                        "model_path": str((root / original["model_path"]).resolve()),
                        "validation_metrics": {**metrics, **marker}, "monthly_metrics": monthly})
    try:
        from xgboost import XGBClassifier
    except Exception:
        raise Phase2Error("XGBoost no pudo cargar; verifica el entorno ML y libomp.") from None
    logger = logging.getLogger(__name__)
    for experiment in TARGET_EXPERIMENTS:
        split, features = splits[experiment], EXPERIMENTS[experiment]
        assert split.test is None
        for model, configurations in CONFIGURATIONS.items():
            original = next(e for e in original_entries if (e["experiment"], e["model"]) == (experiment, model))
            for configuration, overrides in configurations.items():
                started = perf_counter()
                logger.info("event=optimization_start experiment=%s model=%s configuration=%s", experiment, model, configuration)
                parameters = {**original["hyperparameters"], **overrides}
                if parameters.get("missing") == "nan":
                    parameters["missing"] = float("nan")
                estimator = (RandomForestClassifier if model == "random_forest" else XGBClassifier)(**parameters)
                pipeline = make_pipeline(estimator, features)
                pipeline.fit(split.train[features], split.train.target.astype(int))
                directory = out / experiment / model / configuration
                directory.mkdir(parents=True)
                model_path = directory / "model.joblib"
                joblib.dump(pipeline, model_path)
                probability = pipeline.predict_proba(split.validation[features])[:, 1]
                metrics = {**binary_metrics(split.validation.target, probability, .5), **marker}
                monthly = monthly_metrics(split.validation, probability)
                original_metrics, original_monthly = references[(experiment, model)]
                comparison = compare_to_original(metrics, monthly, original_metrics, original_monthly)
                entry = {"experiment": experiment, "model": model, "configuration": configuration, "role": "candidate",
                         "features": features, "model_path": str(model_path.relative_to(out)), "model_sha256": digest(model_path),
                         "hyperparameters": json_parameters(estimator.get_params()), "overrides": overrides,
                         "split": split.metadata, "validation_metrics": metrics, "monthly_metrics": monthly,
                         "comparison": comparison, "elapsed_seconds": perf_counter() - started}
                write_json(directory / "validation_metrics.json", metrics)
                write_json(directory / "validation_monthly.json", monthly)
                write_json(directory / "configuration.json", {**marker, "hyperparameters": entry["hyperparameters"],
                           "comparison": comparison, "pending_controls": review["pending_controls"], "limitations": review["limitations"]})
                entries.append(entry)
                logger.info("event=optimization_complete experiment=%s configuration=%s elapsed_seconds=%.1f validation_ap=%.6f delta_ap=%+.6f",
                            experiment, configuration, entry["elapsed_seconds"], metrics["pr_auc_average_precision"],
                            comparison["delta_vs_original"]["pr_auc_average_precision"])
                del pipeline
    if (any(digest(path) != sha for path, sha in protected.items()) or digest(out / "plan.json") != plan_sha or
            source_hash() != code_sha):
        raise Phase2Error("Una fuente, referencia o plan cambió durante la ronda; resultados incompletos.")
    save_tables(out, entries)
    plot_monthly(out, entries)
    metadata = {**plan, "artifact_format_version": 2, "artifact_type": "exploratory_hyperparameter_round",
                "training_mode": "exploratory", "test_status": "reserved_exploratory_evaluation_forbidden",
                "reference_experiment_sha256": protected[root / "experiment.json"],
                "reference_source_code_sha256": reference["source_code_sha256"],
                "reference_validation_metrics_reproduced": True,
                "reference_policy": "Original trusted local models; hashes/versions/contracts/splits/predictions checked; not refitted",
                "source_code_sha256": code_sha, "dataset_sha256": dataset.sha256,
                "predictor_contract": list(PREDICTOR_COLUMNS), "label_availability": labels.identity(),
                "label_availability_summary": labels.summary, "methodological_review": review, "versions": versions(),
                "plan_sha256": plan_sha, "models": entries, "definitive_model": None,
                "warnings": [EXPLORATORY_WARNING, *review["limitations"], *COMMERCIAL_CONTEXT["limitations"],
                             "Repeated validation use introduces selection optimism; no independent confirmation or uncertainty intervals."]}
    write_json(out / "experiment.json", metadata)
    return out
