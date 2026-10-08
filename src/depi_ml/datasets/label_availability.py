"""Contrato auxiliar de marcas registradas; no amplía las 19 predictoras."""

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from depi_ml.analysis.dataset import LocalDataset, digest, write_json
from depi_ml.errors import Phase2Error


LABEL_COLUMNS = ["appointment_id", "target", "outcome", "label_recorded_at", "label_source"]
LABEL_DEFINITION = {
    "0": "raw_flow.laser_gen.LaserGCloseDate AT TIME ZONE 'America/Lima'",
    "1": "analytics.appointment_outcomes.outcome_resolved_at",
}
LABEL_SEMANTICS = "Recorded timestamp proxy; requires human validation of actual label availability"
EXPECTED_OUTCOMES = {0: "ATTENDED_COMPLETED", 1: "NO_SHOW"}
EXPECTED_SOURCES = {0: "laser_close_date", 1: "outcome_resolved_at"}
TEMPORAL_VALIDATION_POLICY = {
    "version": "outcome_aware_v2",
    "attendance_critical": "label_recorded_at <= prediction_at",
    "attendance_operational_warning": "prediction_at < label_recorded_at < appointment_at",
    "no_show_critical": "label_recorded_at < appointment_at",
    "snapshot_critical": "label_recorded_at > source_snapshot_at",
}


def label_temporal_masks(target, recorded, prediction, appointment):
    """Una única regla por resultado para validación y particiones; NaT no se imputa."""
    return {
        "attendance_recorded_at_or_before_prediction": (target == 0) & (recorded <= prediction),
        "no_show_recorded_before_appointment": (target == 1) & (recorded < appointment),
        "attendance_recorded_between_prediction_and_appointment": (
            (target == 0) & (recorded > prediction) & (recorded < appointment)),
    }


@dataclass
class LabelAvailability:
    frame: pd.DataFrame
    source: Path
    manifest: dict
    sha256: str
    manifest_sha256: str
    summary: dict

    def identity(self):
        return {"sha256": self.sha256, "manifest_sha256": self.manifest_sha256,
                "source_dataset_sha256": self.manifest["source_dataset_sha256"],
                "rows": len(self.frame), "columns": LABEL_COLUMNS,
                "definition": LABEL_DEFINITION, "semantics": LABEL_SEMANTICS}


def aware_timestamp(value):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError
        return stamp.tz_convert("UTC")
    except (ValueError, TypeError):
        raise Phase2Error("Manifest de etiquetas con fecha ausente, inválida o sin zona horaria.") from None


def read_label_csv(path: Path) -> pd.DataFrame:
    """CSV estricto, BIGINT sin paso intermedio por float y fechas UTC tipadas."""
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream, strict=True)
            if next(rows, None) != LABEL_COLUMNS:
                raise Phase2Error("Columnas del auxiliar distintas del contrato de etiquetas.")
            for row in rows:
                if len(row) != len(LABEL_COLUMNS):
                    raise Phase2Error("Registro auxiliar con cantidad de columnas incorrecta.")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        for name in ["appointment_id", "target"]:
            if not frame[name].str.fullmatch(r"-?\d+").all():
                raise Phase2Error(f"Identificador/entero ausente o inválido en {name} del auxiliar.")
            frame[name] = frame[name].astype("int64")
        raw = frame.label_recorded_at.replace("", pd.NA)
        if not raw.dropna().str.contains(r"(?:Z|[+-]\d{2}(?::?\d{2})?)$", regex=True).all():
            raise Phase2Error("label_recorded_at debe tener zona horaria explícita.")
        parsed = pd.to_datetime(raw, utc=True, format="ISO8601", errors="coerce")
        if (raw.notna() & parsed.isna()).any():
            raise Phase2Error("label_recorded_at contiene fechas inválidas o no finitas.")
        frame["label_recorded_at"] = parsed
        return frame
    except (csv.Error, UnicodeError, pd.errors.ParserError, OverflowError, ValueError) as error:
        if isinstance(error, Phase2Error):
            raise
        raise Phase2Error("Formato o tipo inválido en el archivo auxiliar; no se muestran registros.") from None


def validate_label_frame(frame, dataset: LocalDataset, snapshot_at, *, strict_dates=True) -> dict:
    source = dataset.frame
    if source.appointment_id.isna().any() or source.appointment_id.duplicated().any() or not source.target.isin([0, 1]).all():
        raise Phase2Error("CSV original con identificadores duplicados/ausentes o target inválido.")
    if source[["prediction_at", "appointment_at"]].isna().any().any() or (source.prediction_at >= source.appointment_at).any():
        raise Phase2Error("Fechas inconsistentes en el CSV original.")
    if len(frame) != len(source) or frame.empty:
        raise Phase2Error("Cantidad de registros del auxiliar distinta del CSV original.")
    if frame.appointment_id.isna().any() or frame.appointment_id.duplicated().any():
        raise Phase2Error("Identificadores de cita duplicados o ausentes en el auxiliar.")
    if not frame.target.isin([0, 1]).all():
        raise Phase2Error("Tipos de resultado/target inválidos en el auxiliar.")
    if not set(frame.appointment_id) == set(source.appointment_id):
        raise Phase2Error("Identificadores del auxiliar distintos del CSV original.")
    joined = source[["appointment_id", "target", "appointment_at", "prediction_at"]].merge(
        frame, on="appointment_id", how="left", validate="one_to_one", suffixes=("_csv", "_labels"))
    if not joined.target_csv.eq(joined.target_labels).all():
        raise Phase2Error("Resultados del auxiliar distintos del target del CSV original.")
    if not joined.outcome.eq(joined.target_csv.map(EXPECTED_OUTCOMES)).all():
        raise Phase2Error("Outcome incompatible con las clases del CSV original.")
    if not joined.label_source.eq(joined.target_csv.map(EXPECTED_SOURCES)).all():
        raise Phase2Error("Origen de label_recorded_at incompatible con el target.")
    before_appointment = joined.label_recorded_at < joined.appointment_at
    masks = label_temporal_masks(joined.target_csv, joined.label_recorded_at,
                                 joined.prediction_at, joined.appointment_at)
    snapshot = aware_timestamp(snapshot_at)
    future = joined.label_recorded_at > snapshot
    critical = (masks["attendance_recorded_at_or_before_prediction"] |
                masks["no_show_recorded_before_appointment"] | future)
    # Una fila posterior a la instantánea sigue siendo crítica, incluso si su cierre es temprano.
    operational = masks["attendance_recorded_between_prediction_and_appointment"] & ~critical
    if strict_dates and critical.any():
        raise Phase2Error(
            f"Fechas inconsistentes: {int(critical.sum())} etiquetas con anomalías críticas "
            "(asistencia anterior/simultánea a reserva, no-show antes de cita o fecha posterior "
            "a la instantánea PostgreSQL); no se corrigen.")
    original_export = aware_timestamp(dataset.manifest.get("exported_at"))
    summary = {
        "rows": len(joined), "identifiers_match": True, "targets_match": True,
        "duplicate_identifiers": 0, "invalid_outcomes": 0,
        "temporal_validation_policy": TEMPORAL_VALIDATION_POLICY,
        "inconsistent_dates": int(critical.sum()),
        "inconsistent_dates_semantics": "Critical temporal rows only; operational warnings do not block",
        "critical_temporal_errors": {
            "rows": int(critical.sum()),
            "attendance_recorded_at_or_before_prediction": int(masks["attendance_recorded_at_or_before_prediction"].sum()),
            "no_show_recorded_before_appointment": int(masks["no_show_recorded_before_appointment"].sum()),
            "recorded_after_source_snapshot": int(future.sum()),
        },
        "operational_warnings": {
            "rows": int(operational.sum()),
            "attendance_recorded_between_prediction_and_appointment": int(operational.sum()),
            "action": "Retain rows; apply ordinary label_recorded_at cutoffs; require human review",
            "interpretation": "Manual session closure may involve tests or administrative errors; individual causes are not established",
        },
        "recorded_before_appointment": int(before_appointment.sum()),
        "recorded_after_source_snapshot": int(future.sum()),
        "missing_label_recorded_at": int(joined.label_recorded_at.isna().sum()),
        "recorded_at_or_after_dataset_export": int((joined.label_recorded_at >= original_export).sum()),
        "classes": {}, "methodological_status": "pending_human_review",
        "warning": LABEL_SEMANTICS,
    }
    for target in [0, 1]:
        part = joined[joined.target_csv == target]
        lag = (part.label_recorded_at - part.appointment_at).dt.total_seconds() / 86400
        summary["classes"][str(target)] = {
            "rows": len(part), "missing": int(part.label_recorded_at.isna().sum()),
            "critical_temporal_rows": int((critical & (joined.target_csv == target)).sum()),
            "operational_warning_rows": int((operational & (joined.target_csv == target)).sum()),
            "recorded_before_appointment": int((part.label_recorded_at < part.appointment_at).sum()),
            "recorded_after_source_snapshot": int((part.label_recorded_at > snapshot).sum()),
            "recorded_at_or_after_dataset_export": int((part.label_recorded_at >= original_export).sum()),
            "lag_days_median": float(lag.median()) if lag.notna().any() else None,
            "lag_days_p99": float(lag.quantile(.99)) if lag.notna().any() else None,
            "lag_days_max": float(lag.max()) if lag.notna().any() else None,
        }
    summary["status"] = ("blocked_inconsistent_dates" if summary["inconsistent_dates"] else
                         "validated_with_operational_warnings" if operational.any() else
                         "validated_with_missing_labels" if summary["missing_label_recorded_at"] else "validated")
    return summary


def load_label_availability(dataset, labels_path: Path, manifest_path: Path | None = None, *, allow_inconsistent_dates=False):
    path = Path(labels_path)
    manifest_path = Path(manifest_path) if manifest_path else path.with_name(path.stem + "_manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise Phase2Error("Falta el archivo auxiliar de etiquetas o su manifest; usa export-label-availability.")
    try:
        manifest_sha = digest(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sha = digest(path)
        if not isinstance(manifest, dict) or manifest.get("artifact_type") != "label_availability" or manifest.get("format_version") != 1:
            raise Phase2Error("Manifest auxiliar incompatible.")
        if manifest.get("source_dataset_sha256") != dataset.sha256:
            raise Phase2Error("El auxiliar pertenece a otro CSV original (SHA-256 distinto).")
        if manifest.get("source_manifest_sha256") != dataset.manifest_sha256:
            raise Phase2Error("El auxiliar pertenece a otro manifest original (SHA-256 distinto).")
        if manifest.get("columns") != LABEL_COLUMNS or manifest.get("column_count") != len(LABEL_COLUMNS):
            raise Phase2Error("Contrato del manifest auxiliar incorrecto.")
        if manifest.get("sha256") != sha or manifest.get("size_bytes") != path.stat().st_size:
            raise Phase2Error("SHA-256 o tamaño del archivo auxiliar distinto de su manifest.")
        if manifest.get("definition") != LABEL_DEFINITION or manifest.get("semantics") != LABEL_SEMANTICS:
            raise Phase2Error("Definición de fechas registrada incompatible con la extracción requerida.")
        if aware_timestamp(manifest.get("source_dataset_exported_at")) != aware_timestamp(dataset.manifest.get("exported_at")):
            raise Phase2Error("Fecha de exportación original distinta en el manifest auxiliar.")
        aware_timestamp(manifest.get("exported_at"))
        frame = read_label_csv(path)
        if manifest.get("row_count") != len(frame):
            raise Phase2Error("Cantidad de registros distinta del manifest auxiliar.")
        summary = validate_label_frame(frame, dataset, manifest.get("source_snapshot_at"), strict_dates=not allow_inconsistent_dates)
        if digest(path) != sha or digest(manifest_path) != manifest_sha:
            raise Phase2Error("El auxiliar o su manifest cambió durante la validación.")
        return LabelAvailability(frame, path.resolve(), manifest, sha, manifest_sha, summary)
    except (json.JSONDecodeError, TypeError):
        raise Phase2Error("Manifest de disponibilidad de etiquetas inválido.") from None


def attach_labels(dataset, labels):
    """Unión exacta por cita; solo agrega una fecha de auditoría en memoria."""
    return dataset.frame.merge(labels.frame[["appointment_id", "label_recorded_at"]],
                               on="appointment_id", how="left", validate="one_to_one", sort=False)


def write_review_template(path, dataset, labels=None):
    from depi_ml.analysis.audit import REVIEW_REQUIREMENTS
    template = {"dataset_sha256": dataset.sha256, "feature_versions": dataset.manifest["feature_versions"],
                "label_availability_sha256": labels.sha256 if labels else None,
                "label_availability_manifest_sha256": labels.manifest_sha256 if labels else None,
                "reviewer": "", "reviewed_at": "",
                "checks": {name: {"status": "pending", "evidence": "", "instruction": instruction}
                           for name, instruction in REVIEW_REQUIREMENTS.items()}}
    write_json(path, template)
