"""Reproduce RF/XGB originales solo en validación; no fit, SQL ni scores individuales.

La prueba reservada no se materializa. Se exige pin confiable del experimento.
Genera métricas, calibración y subgrupos agregados en un destino nuevo bajo reports/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from depi_ml.analysis.dataset import digest, load_dataset, write_json
from depi_ml.datasets.label_availability import attach_labels, load_label_availability
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.evaluation.metrics import binary_metrics
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerifier
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.sklearn_predictor import positive_class_index
from depi_ml.training.preprocessing import category_value
from depi_ml.training.splits import temporal_split


def calibration(y, probability):
    """Diez bins uniformes fijados, sin ajustar calibrador ni bandas de riesgo."""
    cells = []
    for index in range(10):
        mask = (probability >= index / 10) & (
            (probability < (index + 1) / 10) if index < 9 else (probability <= 1))
        if mask.any():
            cells.append({"lower": index / 10, "upper": (index + 1) / 10,
                          "rows": int(mask.sum()), "positives": int(y[mask].sum()),
                          "mean_probability": float(probability[mask].mean()),
                          "observed_rate": float(y[mask].mean())})
    return {"bin_policy": "10 fixed uniform bins; final bin includes 1",
            "cells": cells, "rows": len(y), "mean_probability": float(probability.mean()),
            "observed_rate": float(y.mean()),
            "mean_probability_minus_observed": float(probability.mean() - y.mean()),
            "ece_uniform_10": sum(cell["rows"] / len(y) * abs(
                cell["mean_probability"] - cell["observed_rate"]) for cell in cells),
            "interpretation": "descriptive, dependent on bins and population; not operational calibration approval"}


def verify_metrics(current, recorded):
    for name in ("rows", "no_show", "confusion_matrix"):
        if current[name] != recorded[name]:
            raise ValueError("Stored validation counts not reproduced")
    for name in ("prevalence", "roc_auc", "pr_auc_average_precision", "precision", "recall", "f1", "brier_score", "threshold"):
        if not np.isclose(current[name], recorded[name], rtol=1e-12, atol=1e-12):
            raise ValueError("Stored validation metrics not reproduced")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--csv", type=Path, default=Path("data/exports/training_dataset_v1.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/exports/training_dataset_v1_labels.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path.cwd().resolve() / "reports") or output.exists():
        parser.error("Destino nuevo dentro de reports/ requerido.")
    ArtifactVerifier(args.reference, args.manifest_sha256).verify()
    manifest = json.loads((args.reference / "experiment.json").read_text())
    if digest(args.reference / "experiment.json") != args.manifest_sha256:
        raise ValueError("Trusted manifest changed")
    dataset = load_dataset(args.csv)
    labels = load_label_availability(dataset, args.labels)
    if dataset.sha256 != manifest["dataset_sha256"] or labels.identity() != manifest["label_availability"]:
        raise ValueError("Dataset/labels differ from original experiment")
    entries = {e["model"]: e for e in manifest["models"] if e["experiment"] == "full"}
    recorded = entries["random_forest"]["split"]
    split = temporal_split(attach_labels(dataset, labels), recorded["validation_start"],
                           recorded["test_start"], recorded["test_end"],
                           dataset.manifest["exported_at"], include_test=False)
    if split.test is not None or split.metadata != recorded:
        raise ValueError("Original temporal split not reproduced")
    registry = ModelRegistry.load(ApiSettings(model_reference_dir=args.reference,
                                             model_manifest_sha256=args.manifest_sha256))
    validation = split.validation
    y = validation.target.to_numpy(dtype=int)
    month = validation.prediction_at.dt.tz_convert("America/Lima").dt.strftime("%Y-%m").to_numpy()
    report = {"status": "EXPLORATORY_NOT_VALIDATED", "training_status": "TRAINING_REUSED_EXISTING",
              "test_materialized": False, "test_predictions": 0, "new_model_fits": 0,
              "reference": str(args.reference), "reference_sha256": args.manifest_sha256,
              "dataset_sha256": dataset.sha256, "label_availability": labels.identity(),
              "features": PREDICTOR_COLUMNS, "split": split.metadata, "models": []}
    # Baseline original de prior: identidad y prevalencia comprobadas en su manifest.
    prior = float(split.train.target.mean())
    baseline_probability = np.full(len(validation), prior)
    baseline_metrics = binary_metrics(y, baseline_probability, .5)
    verify_metrics(baseline_metrics, entries["baseline"]["validation_metrics"])
    report["baseline"] = {"source": "original baseline prior, reproduced algebraically without fit",
                          "training_prevalence": prior, "metrics": baseline_metrics}
    for info in registry.list_models():
        pipeline = registry.get(info.model_key).pipeline
        probability = pipeline.predict_proba(validation[PREDICTOR_COLUMNS])[:, positive_class_index(pipeline)]
        metrics = binary_metrics(y, probability, .5)
        verify_metrics(metrics, entries[info.algorithm]["validation_metrics"])
        metrics["log_loss"] = float(log_loss(y, probability, labels=[0, 1]))
        monthly = []
        for value in sorted(set(month)):
            mask = month == value
            monthly.append({"month": value, "date_basis": "prediction_at America/Lima",
                            **binary_metrics(y[mask], probability[mask], .5),
                            "calibration": calibration(y[mask], probability[mask])})
        clinics = []
        for clinic in sorted(validation.clinic_id.unique()):
            mask = validation.clinic_id.to_numpy() == clinic
            if mask.sum() >= 30:
                clinics.append({"clinic_id": int(clinic), **binary_metrics(y[mask], probability[mask], .5)})
        unknown = {}
        transformer = pipeline.named_steps["preprocess"]
        columns = next(columns for name, _, columns in transformer.transformers_ if name == "category")
        categories = transformer.named_transformers_["category"].named_steps["encoder"].categories_
        for column, known in zip(columns, categories, strict=True):
            values = validation[column].map(lambda v: category_value(v, column in
                {"is_fwa", "has_medical_evaluation", "has_type4_service"})).fillna("__MISSING__")
            unknown[column] = int((~values.isin(known)).sum())
        report["models"].append({"model_key": info.model_key, "model_sha256": info.model_sha256,
                                 "model_version": info.model_version, "metrics": metrics,
                                 "metrics_reproduced": True, "monthly": monthly, "clinics": clinics,
                                 "unknown_validation_categories": unknown,
                                 "calibration": calibration(y, probability),
                                 "delta_vs_baseline": {k: metrics[k] - baseline_metrics[k]
                                     for k in ("roc_auc", "pr_auc_average_precision", "brier_score")}})
    if (digest(dataset.source) != dataset.sha256 or digest(labels.source) != labels.sha256
            or digest(dataset.manifest_source) != dataset.manifest_sha256
            or digest(args.labels.with_name(args.labels.stem + "_manifest.json")) != labels.manifest_sha256
            or digest(args.reference / "experiment.json") != args.manifest_sha256):
        raise ValueError("Original source changed during verification")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics_summary.json", report)
    lines = ["# Validación reproducida", "", "EXPLORATORY_NOT_VALIDATED. Umbral diagnóstico 0.5.", "",
             "Sin test ni fit. Cortes originales, purga por label_recorded_at y exactamente 19 features.", "",
             "| Modelo | N | Positivos | ROC-AUC | AP | Precision | Recall | F1 | Brier |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in [{"model_key": "baseline", "metrics": baseline_metrics}, *report["models"]]:
        m = row["metrics"]
        lines.append("| " + " | ".join([row["model_key"], str(m["rows"]), str(m["no_show"]),
                     *[f"{m[k]:.6f}" for k in ("roc_auc", "pr_auc_average_precision", "precision", "recall", "f1", "brier_score")]]) + " |")
    (output / "metrics_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "validation_rows": len(validation),
                      "models_reproduced": len(report["models"]), "test_predictions": 0, "new_model_fits": 0}))


if __name__ == "__main__":
    main()
