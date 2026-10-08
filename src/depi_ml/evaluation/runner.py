"""Carga solo artefactos locales propios y verifica dataset, hash y particiones."""

import json
from pathlib import Path

import joblib

from depi_ml.analysis.audit import audit_dataset
from depi_ml.analysis.dataset import Phase2Error, digest, load_dataset, new_output, write_json
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.datasets.label_availability import attach_labels, load_label_availability
from depi_ml.evaluation.reports import evaluate_partition
from depi_ml.training.experiments import EXPERIMENTS, check_options, experiment_frame, source_hash, versions
from depi_ml.training.review import require_review
from depi_ml.training.splits import temporal_split, utc_timestamp


def evaluate(csv_path, manifest_path, experiment_path, output, explain=False, *, labels_path=None, labels_manifest_path=None):
    dataset = load_dataset(csv_path, manifest_path)
    root = Path(experiment_path).resolve()
    try:
        metadata = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise Phase2Error("Experimento local incompleto o inexistente.") from None
    if metadata.get("artifact_format_version") != 2 or metadata.get("dataset_sha256") != dataset.sha256:
        raise Phase2Error("Versión de artefacto o hash del dataset incompatible.")
    if metadata.get("predictor_contract") != PREDICTOR_COLUMNS:
        raise Phase2Error("Contrato predictor distinto del experimento.")
    if metadata.get("versions") != versions():
        raise Phase2Error("Versiones distintas del entorno de entrenamiento; recrea el entorno registrado.")
    if metadata.get("source_code_sha256") != source_hash():
        raise Phase2Error("Código distinto del registrado en entrenamiento; restaura la versión original para evaluar.")
    if labels_path is None:
        raise Phase2Error("Evaluación requiere --labels con el mismo auxiliar del entrenamiento.")
    labels = load_label_availability(dataset, labels_path, labels_manifest_path)
    if metadata.get("label_availability") != labels.identity():
        raise Phase2Error("El auxiliar de disponibilidad o su manifest difiere del utilizado para entrenar.")
    frame = attach_labels(dataset, labels)
    require_review(dataset, audit_dataset(dataset), root / "review.json", labels)
    threshold, min_group, samples, seed = (metadata[k] for k in ["threshold", "min_group", "importance_samples", "seed"])
    check_options(threshold, min_group, samples)
    expected = {(name, model) for name in EXPERIMENTS for model in ["baseline", "random_forest", "xgboost"]}
    if {(e.get("experiment"), e.get("model")) for e in metadata["models"]} != expected or len(metadata["models"]) != len(expected):
        raise Phase2Error("Faltan modelos o experimentos obligatorios.")
    # Verificar todo antes de deserializar o producir métricas.
    prepared = []
    for entry in metadata["models"]:
        name, recorded = entry["experiment"], entry["split"]
        if utc_timestamp(recorded["labels_observed_until"]) != utc_timestamp(dataset.manifest["exported_at"]):
            raise Phase2Error("Política de disponibilidad distinta de la revisión/manifest.")
        if entry["features"] != EXPERIMENTS[name]:
            raise Phase2Error("Variables distintas del diseño de sensibilidad.")
        model_path = (root / entry["model_path"]).resolve()
        if root not in model_path.parents or not model_path.is_file() or digest(model_path) != entry["model_sha256"]:
            raise Phase2Error("Ruta o hash de modelo inválido.")
        split = temporal_split(experiment_frame(frame, name), recorded["validation_start"], recorded["test_start"],
                               recorded["test_end"], recorded["labels_observed_until"])
        if split.metadata != recorded:
            raise Phase2Error("Particiones recalculadas distintas del experimento.")
        prepared.append((entry, model_path, split.test))
    out = new_output(output, dataset)
    results = []
    for entry, model_path, test in prepared:
        pipeline = joblib.load(model_path)
        if pipeline.named_steps["contract"].features_ != entry["features"]:
            raise Phase2Error("Contrato del modelo persistido incompatible.")
        metrics = evaluate_partition(pipeline, test, out / entry["experiment"] / entry["model"], entry["features"],
                  threshold, min_group, samples, seed, explain=explain)
        results.append({"experiment": entry["experiment"], "model": entry["model"], **metrics})
    write_json(out / "test_comparison.json", {"dataset_sha256": dataset.sha256,
               "label_availability": labels.identity(),
               "status": "experimental_test_evaluation_not_production_approval", "versions": versions(),
               "threshold": threshold, "results": results,
               "warning": "No ajustar features, hiperparámetros ni umbrales tras observar esta prueba; reservar un nuevo período si se itera."})
    return out
