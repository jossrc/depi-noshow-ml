"""Entrenamiento reproducible; prueba reservada para evaluate."""

import importlib.metadata
import math
from pathlib import Path
import platform
import hashlib
import csv
import logging
from time import perf_counter

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from depi_ml.analysis.audit import audit_dataset
from depi_ml.analysis.dataset import Phase2Error, digest, load_dataset, new_output, write_json
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.datasets.label_availability import attach_labels, load_label_availability
from depi_ml.evaluation.reports import evaluate_partition
from depi_ml.training.preprocessing import CATALOG_COLUMNS, make_pipeline
from depi_ml.training.review import EXPLORATORY_WARNING, exploratory_review, require_review
from depi_ml.training.splits import temporal_split, utc_timestamp


EXPERIMENTS = {
    "full": list(PREDICTOR_COLUMNS),
    "without_is_fwa": [c for c in PREDICTOR_COLUMNS if c != "is_fwa"],
    "without_catalog": [c for c in PREDICTOR_COLUMNS if c not in CATALOG_COLUMNS],
    "without_uid_minus_one": list(PREDICTOR_COLUMNS),
}


def experiment_frame(frame, name):
    if name == "without_uid_minus_one":
        return frame.loc[~((frame.target == 1) & (frame.no_show_uid_minus_one == 1))].copy()
    return frame


def versions():
    return {"python": platform.python_version(), **{package: importlib.metadata.version(package)
             for package in ["depi-noshow-ml", "pandas", "numpy", "scikit-learn", "xgboost", "matplotlib", "joblib"]}}


def source_hash():
    root = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        sha.update(str(path.relative_to(root)).encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def json_parameters(parameters):
    return {name: (str(value) if isinstance(value, float) and not math.isfinite(value) else value)
            for name, value in parameters.items()}


def check_options(threshold, min_group, importance_samples):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise Phase2Error("El umbral diagnóstico debe estar entre 0 y 1.")
    if min_group < 10 or importance_samples <= 0:
        raise Phase2Error("Grupos >=10 e importance_samples positivo son obligatorios.")


def write_validation_comparison(out, metadata):
    entries = [{"experiment": e["experiment"], "model": e["model"], **e["validation_metrics"]}
               for e in metadata["models"]]
    write_json(out / "validation_comparison.json", entries)
    columns = ["experiment", "model", "rows", "roc_auc", "pr_auc_average_precision",
               "recall", "precision", "f1", "brier_score", "threshold"]
    with (out / "validation_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*columns, "status", "evaluation_partition"])
        writer.writeheader()
        writer.writerows({**{key: entry[key] for key in columns}, "status": metadata["status"],
                         "evaluation_partition": "validation"} for entry in entries)
    lines = [metadata["status"], "", f"Validación temporal; semilla {metadata['seed']}; umbral diagnóstico {metadata['threshold']}.",
             "No se evaluó prueba ni se seleccionó un modelo definitivo.", "",
             "| Experimento | Modelo | ROC-AUC | PR-AUC (AP) | Recall | Precision | F1 | Brier |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for entry in entries:
        values = [entry[key] for key in ["roc_auc", "pr_auc_average_precision", "recall", "precision", "f1", "brier_score"]]
        lines.append("| " + " | ".join([entry["experiment"], entry["model"],
                     *[f"{value:.4f}" if value is not None else "N/A" for value in values]]) + " |")
    if metadata.get("methodological_review"):
        lines += ["", "Limitaciones metodológicas:", "",
                  *[f"- {value}" for value in metadata["methodological_review"]["limitations"]]]
    (out / "validation_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(csv_path, manifest_path, output, review_path, validation_start, test_start, test_end,
          seed=42, threshold=.5, min_group=30, importance_samples=2000, *, labels_path=None, labels_manifest_path=None,
          exploratory=False):
    check_options(threshold, min_group, importance_samples)
    dataset = load_dataset(csv_path, manifest_path)
    if labels_path is None:
        raise Phase2Error("Entrenamiento requiere --labels con el auxiliar validado; no se permite demora global.")
    labels = load_label_availability(dataset, labels_path, labels_manifest_path)
    frame = attach_labels(dataset, labels)
    audit = audit_dataset(dataset, descriptive_before=utc_timestamp(test_start) if exploratory else None)
    audit["label_availability"] = labels.summary
    review = (exploratory_review(dataset, audit, review_path, labels) if exploratory else
              require_review(dataset, audit, review_path, labels))
    if exploratory:
        audit["training_status"] = EXPLORATORY_WARNING
        audit["pending_review"] = review["pending_controls"]
        audit["exploratory_limitations"] = review["limitations"]
    splits = {}
    for name in EXPERIMENTS:
        splits[name] = temporal_split(experiment_frame(frame, name), validation_start, test_start, test_end,
                                      dataset.manifest["exported_at"], include_test=not exploratory)
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise Phase2Error("Instala XGBoost con '.[ml]'.") from None
    except Exception:
        raise Phase2Error("XGBoost no pudo cargar su biblioteca nativa; en macOS verifica libomp (consulta docs/phase2.md).") from None
    # Ningún modelo se genera si falla una partición de alguno de los experimentos obligatorios.
    out = new_output(output, dataset)
    write_json(out / "audit.json", audit)
    write_json(out / "review.json", review)
    metadata = {
        "artifact_format_version": 2,
        "status": EXPLORATORY_WARNING if exploratory else "development_experiment_not_production",
        "training_mode": "exploratory" if exploratory else "reviewed",
        "modeling_partitions": ["train", "validation"],
        "methodological_review": review if exploratory else None,
        "dataset_name": dataset.manifest["dataset_name"], "dataset_sha256": dataset.sha256,
        "dataset_rows": len(dataset.frame), "feature_versions": dataset.manifest["feature_versions"],
        "predictor_contract": list(PREDICTOR_COLUMNS), "seed": seed, "versions": versions(),
        "source_code_sha256": source_hash(), "threshold": threshold,
        "label_availability": labels.identity(), "label_availability_summary": labels.summary,
        "threshold_policy": "Fixed diagnostic threshold; not tuned on test; no LOW/MEDIUM/HIGH bands",
        "min_group": min_group, "importance_samples": importance_samples, "models": [],
        "test_status": "reserved_exploratory_evaluation_forbidden" if exploratory else "reserved_for_evaluate",
        "hyperparameter_selection": "fixed a priori; no search on test",
        "warnings": ["label_recorded_at is a recorded timestamp proxy; actual availability requires external human validation.",
                     "Missing and late administrative closes are excluded by cutoff; assess differential selection by target.",
                     "Excluding UID -1 rows does not rebuild historical features: previous_no_show may still count UID -1 events.",
                     "Population conditioned on original eligibility/exclusion criteria; not validated for all future bookings."],
    }
    context = None
    if exploratory:
        metadata["warnings"] = [EXPLORATORY_WARNING, *review["limitations"], *metadata["warnings"]]
        context = {"status": EXPLORATORY_WARNING, "evaluation_partition": "validation",
                   "pending_methodological_controls": review["pending_controls"],
                   "limitations": review["limitations"]}
    logger = logging.getLogger(__name__)
    for name, features in EXPERIMENTS.items():
        split = splits[name]
        negatives = int((split.train.target == 0).sum())
        positives = int((split.train.target == 1).sum())
        estimators = {
            "baseline": DummyClassifier(strategy="prior", random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                              class_weight="balanced_subsample", random_state=seed, n_jobs=2),
            "xgboost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05,
                       subsample=.8, colsample_bytree=.8, objective="binary:logistic", eval_metric="logloss",
                       tree_method="hist", scale_pos_weight=negatives / positives, random_state=seed, n_jobs=2),
        }
        for model_name, estimator in estimators.items():
            started = perf_counter()
            logger.info("event=model_training_start experiment=%s model=%s train_rows=%d validation_rows=%d mode=%s",
                        name, model_name, len(split.train), len(split.validation), metadata["training_mode"])
            directory = out / name / model_name
            directory.mkdir(parents=True)
            pipeline = make_pipeline(estimator, features)
            pipeline.fit(split.train.loc[:, features], split.train.target.astype(int))
            model_path = directory / "model.joblib"
            joblib.dump(pipeline, model_path)
            metrics = evaluate_partition(pipeline, split.validation, directory / "validation", features,
                      threshold, min_group, importance_samples, seed, report_context=context)
            entry = {"experiment": name, "model": model_name, "features": features,
                     "model_path": str(model_path.relative_to(out)), "model_sha256": digest(model_path),
                     "hyperparameters": json_parameters(estimator.get_params()), "split": split.metadata,
                     "validation_metrics": metrics}
            metadata["models"].append(entry)
            logger.info("event=model_training_complete experiment=%s model=%s elapsed_seconds=%.1f validation_ap=%.6f",
                        name, model_name, perf_counter() - started, metrics["pr_auc_average_precision"])
    # Integridad de las fuentes también al terminar un entrenamiento potencialmente largo.
    if (digest(dataset.source) != dataset.sha256 or digest(dataset.manifest_source) != dataset.manifest_sha256 or
            digest(labels.source) != labels.sha256 or
            digest(labels.source.with_name(labels.source.stem + "_manifest.json") if labels_manifest_path is None
                   else Path(labels_manifest_path)) != labels.manifest_sha256):
        raise Phase2Error("Los archivos fuente cambiaron durante el entrenamiento; experimento incompleto.")
    write_validation_comparison(out, metadata)
    write_json(out / "experiment.json", metadata)  # Marca de finalización: solo tras todos los modelos.
    return out
