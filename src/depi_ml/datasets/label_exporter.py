"""Extracción COPY TO en la instantánea READ ONLY existente, sin ETL ni DDL."""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

from psycopg import sql

from depi_ml.analysis.dataset import digest, load_dataset, write_json
from depi_ml.config import Settings
from depi_ml.datasets.exporter import publish_artifacts, stream_copy
from depi_ml.datasets.label_availability import (
    LABEL_COLUMNS, LABEL_DEFINITION, LABEL_SEMANTICS, read_label_csv, validate_label_frame,
)
from depi_ml.datasets.validator import inspect_dataset
from depi_ml.db.connection import read_only_connection
from depi_ml.errors import Phase2Error


REQUIRED_TYPES = {
    ("analytics", "appointment_outcomes", "appointment_id"): "int8",
    ("analytics", "appointment_outcomes", "client_id"): "int8",
    ("analytics", "appointment_outcomes", "clinic_id"): "int4",
    ("analytics", "appointment_outcomes", "appointment_at"): "timestamptz",
    ("analytics", "appointment_outcomes", "diary_laser_id"): "int8",
    ("analytics", "appointment_outcomes", "outcome"): "appointment_outcome_enum",
    ("analytics", "appointment_outcomes", "outcome_resolved_at"): "timestamptz",
    ("raw_flow", "laser_gen", "LaserGID"): "int8",
    ("raw_flow", "laser_gen", "LaserGClientID"): "int8",
    ("raw_flow", "laser_gen", "LaserGClinicID"): "int4",
    ("raw_flow", "laser_gen", "LaserGCloseDate"): "timestamp",
}

JOINS = """
FROM analytics.appointment_training_dataset_v1 d
LEFT JOIN analytics.appointment_outcomes o ON o.appointment_id = d.appointment_id
LEFT JOIN raw_flow.laser_gen l ON d.target = 0 AND l."LaserGID" = o.diary_laser_id
"""


def validate_sources(connection):
    rows = connection.execute("""
        SELECT table_schema, table_name, column_name, udt_name
        FROM information_schema.columns
        WHERE (table_schema = 'analytics' AND table_name = 'appointment_outcomes')
           OR (table_schema = 'raw_flow' AND table_name = 'laser_gen')
    """).fetchall()
    types = {(r["table_schema"], r["table_name"], r["column_name"]): r["udt_name"] for r in rows}
    if any(types.get(name) != kind for name, kind in REQUIRED_TYPES.items()):
        raise Phase2Error("Faltan columnas fuente o sus tipos difieren del contrato; no se convierte implícitamente la zona horaria.")
    counts = connection.execute("""
        SELECT COUNT(*) FILTER (WHERE o.appointment_id IS NULL) AS missing_outcomes,
          COUNT(*) FILTER (WHERE o.client_id IS DISTINCT FROM d.client_id
            OR o.clinic_id IS DISTINCT FROM d.clinic_id
            OR o.appointment_at IS DISTINCT FROM d.appointment_at) AS outcome_identity_mismatch,
          COUNT(*) FILTER (WHERE (d.target = 0 AND o.outcome::text IS DISTINCT FROM 'ATTENDED_COMPLETED')
            OR (d.target = 1 AND o.outcome::text IS DISTINCT FROM 'NO_SHOW')) AS outcome_type_mismatch,
          COUNT(*) FILTER (WHERE d.target = 0 AND l."LaserGID" IS NULL) AS missing_attendance_session,
          COUNT(*) FILTER (WHERE d.target = 0 AND (l."LaserGClientID" IS DISTINCT FROM d.client_id
            OR l."LaserGClinicID" IS DISTINCT FROM d.clinic_id)) AS session_identity_mismatch
    """ + JOINS).fetchone()
    if any(counts.values()):
        details = ", ".join(f"{name}={int(count)}" for name, count in counts.items() if count)
        raise Phase2Error("Fuentes PostgreSQL inconsistentes con el dataset: " + details + ". Exportación detenida.")


def label_copy_query():
    return sql.SQL("""
        COPY (SELECT d.appointment_id, d.target, o.outcome::text AS outcome,
          replace((CASE WHEN d.target = 0
            THEN l."LaserGCloseDate" AT TIME ZONE 'America/Lima'
            ELSE o.outcome_resolved_at END)::text, ' ', 'T') AS label_recorded_at,
          CASE WHEN d.target = 0 THEN 'laser_close_date'
            ELSE 'outcome_resolved_at' END AS label_source
    """ + JOINS + """
        ORDER BY d.appointment_id)
        TO STDOUT WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8', NULL '')
    """)


def copy_labels(connection, path, buffer_size):
    with path.open("wb", buffering=buffer_size) as stream:
        with connection.cursor() as cursor:
            with cursor.copy(label_copy_query()) as copy:
                for block in copy:
                    stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())


def export_label_availability(settings: Settings, csv_path: Path, manifest_path: Path | None, output: Path):
    dataset = load_dataset(csv_path, manifest_path)
    source_manifest = dataset.manifest_source
    source_manifest_hash = dataset.manifest_sha256
    if digest(source_manifest) != source_manifest_hash:
        raise Phase2Error("El manifest original cambió tras validarse; extracción detenida.")
    if (settings.schema, settings.table) != ("analytics", "appointment_training_dataset_v1"):
        raise Phase2Error("El auxiliar requiere analytics.appointment_training_dataset_v1; verifica configuración sin alterar PostgreSQL.")
    if (dataset.manifest.get("source_schema"), dataset.manifest.get("source_table")) != (settings.schema, settings.table):
        raise Phase2Error("Origen del manifest distinto de la tabla PostgreSQL configurada.")
    output = Path(output).resolve()
    if output.suffix.lower() != ".csv":
        raise Phase2Error("El destino auxiliar debe terminar en .csv.")
    auxiliary_manifest = output.with_name(output.stem + "_manifest.json")
    quality = output.with_name(output.stem + "_quality_report.json")
    protected = {dataset.source, source_manifest.resolve()}
    if any(path in protected or path.exists() for path in [output, auxiliary_manifest, quality]):
        raise Phase2Error("No se sobrescriben el CSV/manifest original ni auxiliares existentes; elige un destino nuevo.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".depi-labels-", dir=output.parent) as directory:
        staging = Path(directory)
        staged_labels = staging / "labels.csv"
        with read_only_connection(settings) as connection:
            transaction = connection.execute("""
                SELECT transaction_timestamp() AS snapshot_at,
                  current_setting('transaction_read_only') AS read_only,
                  current_setting('transaction_isolation') AS isolation
            """).fetchone()
            if transaction["read_only"] != "on" or transaction["isolation"] != "repeatable read":
                raise Phase2Error("La extracción exige REPEATABLE READ READ ONLY.")
            inspect_dataset(connection, settings)
            # Reutilizar COPY de Fase 1 para comparar todos los bytes y las 27 columnas.
            current_source = staging / "source_check.csv"
            stream_copy(connection, settings, current_source, settings.batch_size)
            if digest(current_source) != dataset.sha256:
                raise Phase2Error("PostgreSQL ya no corresponde al CSV original (hash de las 27 columnas distinto). Exportación detenida.")
            validate_sources(connection)
            copy_labels(connection, staged_labels, settings.batch_size)
            frame = read_label_csv(staged_labels)
            summary = validate_label_frame(frame, dataset, transaction["snapshot_at"], strict_dates=False)
        if digest(dataset.source) != dataset.sha256 or digest(source_manifest) != source_manifest_hash:
            raise Phase2Error("El CSV o manifest original cambió durante la extracción; no se publica el auxiliar.")
        manifest = {
            "artifact_type": "label_availability", "format_version": 1,
            "columns": LABEL_COLUMNS, "column_count": len(LABEL_COLUMNS), "row_count": len(frame),
            "sha256": digest(staged_labels), "size_bytes": staged_labels.stat().st_size,
            "source_dataset_sha256": dataset.sha256, "source_manifest_sha256": source_manifest_hash,
            "source_dataset_exported_at": dataset.manifest["exported_at"],
            "source_snapshot_at": transaction["snapshot_at"].isoformat(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "definition": LABEL_DEFINITION, "semantics": LABEL_SEMANTICS,
            "encoding": "UTF-8", "timezone": "UTC", "order_by": ["appointment_id"],
            "sources": ["analytics.appointment_training_dataset_v1", "analytics.appointment_outcomes", "raw_flow.laser_gen"],
            "source_dataset_matches_database": True,
            "transaction": {"read_only": True, "isolation": "REPEATABLE READ"},
            "validation": summary,
        }
        staged_manifest, staged_quality = staging / "manifest.json", staging / "quality.json"
        write_json(staged_manifest, manifest)
        write_json(staged_quality, {"source_dataset_sha256": dataset.sha256, "label_availability_sha256": manifest["sha256"], **summary})
        publish_artifacts([(staged_manifest, auxiliary_manifest), (staged_quality, quality), (staged_labels, output)], staging)
    return output, auxiliary_manifest, quality, summary
