"""Validación local y vista de particiones sin entrenar ni acceder a PostgreSQL."""

from depi_ml.analysis.dataset import Phase2Error, load_dataset, new_output, write_json
from depi_ml.datasets.label_availability import attach_labels, load_label_availability, write_review_template
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.training.splits import temporal_split


def validate_labels(csv_path, manifest_path, labels_path, labels_manifest_path, output,
                    validation_start=None, test_start=None, test_end=None):
    dataset = load_dataset(csv_path, manifest_path)
    labels = load_label_availability(dataset, labels_path, labels_manifest_path, allow_inconsistent_dates=True)
    report = {"dataset_sha256": dataset.sha256, "label_availability": labels.identity(),
              "validation_recomputed_from_files": True,
              "validation": labels.summary, "predictors": list(PREDICTOR_COLUMNS),
              "label_recorded_at_role": "auditoria_no_predictor",
              "training_status": "blocked_inconsistent_dates" if labels.summary["critical_temporal_errors"]["rows"] else "pending_human_review"}
    boundaries = [validation_start, test_start, test_end]
    if any(value is not None for value in boundaries):
        if any(value is None for value in boundaries):
            raise Phase2Error("La vista de particiones requiere los tres cortes temporales.")
        if labels.summary["critical_temporal_errors"]["rows"]:
            report["temporal_split_preview_status"] = "blocked_inconsistent_dates; no rows corrected or removed"
        else:
            partitions = temporal_split(attach_labels(dataset, labels), *boundaries, dataset.manifest["exported_at"], require_both_classes=False)
            report["temporal_split_preview"] = partitions.metadata
            report["partitions_with_insufficient_classes"] = [name for name in ["train", "validation", "test"]
                                                             if getattr(partitions, name).target.nunique() != 2]
    out = new_output(output, dataset)
    write_json(out / "label_availability_report.json", report)
    write_review_template(out / "methodology_review_template.json", dataset, labels)
    return out, report
