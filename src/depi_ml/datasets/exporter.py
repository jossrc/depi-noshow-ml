"""COPY por bloques y publicación de artefactos verificados."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import os
import platform
from pathlib import Path
import shutil
import tempfile

from psycopg import sql
import psycopg

from depi_ml.config import Settings, positive_integer
from depi_ml.db.connection import read_only_connection
from depi_ml.datasets.schema import EXPORT_COLUMNS, TIMESTAMP_COLUMNS
from depi_ml.datasets.validator import inspect_dataset, report_json

logger = logging.getLogger(__name__)


class ExportError(ValueError):
    """El CSV no cumple las verificaciones de integridad."""


@dataclass(frozen=True)
class ExportResult:
    csv_path: Path
    manifest_path: Path
    quality_path: Path
    row_count: int
    size_bytes: int
    attended_count: int
    no_show_count: int


def copy_query(settings: Settings):
    columns = sql.SQL(", ").join(
        sql.SQL("replace({}::text, ' ', 'T') AS {}").format(sql.Identifier(name), sql.Identifier(name))
        if name in TIMESTAMP_COLUMNS else sql.Identifier(name)
        for name in EXPORT_COLUMNS
    )
    return sql.SQL(
        "COPY (SELECT {} FROM {} ORDER BY appointment_at, appointment_id) "
        "TO STDOUT WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8', NULL '')"
    ).format(columns, sql.Identifier(settings.schema, settings.table))


def stream_copy(connection, settings: Settings, path: Path, batch_size: int):
    # COPY decide el tamaño de sus bloques de red. batch_size son bytes del buffer
    # del archivo local; también determina cada cuánto verificar CSV registra avance.
    total_bytes = 0
    next_progress = 16 * 1024 * 1024
    with path.open("wb", buffering=batch_size) as destination:
        with connection.cursor() as cursor:
            with cursor.copy(copy_query(settings)) as copy:
                for block in copy:
                    destination.write(block)
                    total_bytes += len(block)
                    if total_bytes >= next_progress:
                        logger.info("event=copy_progress bytes=%d", total_bytes)
                        next_progress = total_bytes + 16 * 1024 * 1024
        destination.flush()
        os.fsync(destination.fileno())
    logger.info("event=copy_complete bytes=%d", total_bytes)


def verify_csv(path: Path, expected_count: int, batch_size: int = 10000) -> dict:
    count = 0
    classes = {"0": 0, "1": 0}
    target_index = EXPORT_COLUMNS.index("target")
    # csv.reader cuenta registros lógicos: respeta campos entrecomillados y saltos.
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, strict=True)
        if next(reader, None) != EXPORT_COLUMNS:
            raise ExportError("Los encabezados del CSV no coinciden con el contrato.")
        for row in reader:
            if len(row) != len(EXPORT_COLUMNS):
                raise ExportError("El CSV contiene una fila con número de columnas incorrecto.")
            target = row[target_index]
            if target not in classes:
                raise ExportError("El CSV contiene un target inválido.")
            classes[target] += 1
            count += 1
            if count % batch_size == 0:
                logger.info("event=csv_verification rows=%d", count)
    if count != expected_count:
        raise ExportError("El total de registros del CSV difiere del snapshot PostgreSQL.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"row_count": count, "class_counts": classes,
            "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def publish_artifacts(pairs: list[tuple[Path, Path]], staging: Path):
    """CSV al final; rollback de todos los destinos si falla una sustitución."""
    backups = {}
    for index, (_, destination) in enumerate(pairs):
        if destination.exists():
            backup = staging / f"backup-{index}"
            try:
                os.link(destination, backup)
            except OSError:
                shutil.copy2(destination, backup)
            backups[destination] = backup
    published = []
    try:
        for source, destination in pairs:
            os.replace(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            if destination in backups:
                os.replace(backups[destination], destination)
            else:
                destination.unlink(missing_ok=True)
        raise


def export_dataset(settings: Settings, output: Path | None = None,
                   batch_size: int | None = None) -> ExportResult:
    size = settings.batch_size if batch_size is None else batch_size
    positive_integer(str(size), "batch_size")
    csv_path = Path(output) if output is not None else settings.export_dir / "training_dataset_v1.csv"
    if csv_path.suffix.lower() != ".csv":
        raise ExportError("El destino de exportación debe terminar en .csv.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = csv_path.with_name(csv_path.stem + "_manifest.json")
    quality_path = csv_path.with_name(csv_path.stem + "_quality_report.json")
    logger.info("event=export_start")
    # Todos los temporales viven en el mismo filesystem que el destino.
    with tempfile.TemporaryDirectory(prefix=".depi-export-", dir=csv_path.parent) as temporary:
        staging = Path(temporary)
        staged_csv = staging / "dataset.csv"
        with read_only_connection(settings) as connection:
            report = inspect_dataset(connection, settings)
            stats = report["statistics"]
            logger.info("event=validated rows=%d", stats["row_count"])
            stream_copy(connection, settings, staged_csv, size)
            verification = verify_csv(staged_csv, stats["row_count"], size)
            if verification["class_counts"] != {"0": stats["attended_count"], "1": stats["no_show_count"]}:
                raise ExportError("La distribución de clases del CSV difiere de PostgreSQL.")
        # La transacción debe cerrarse correctamente antes de publicar.
        exported_at = datetime.now(timezone.utc)
        report["csv_verification"] = verification
        report["validations"]["csv_row_count_matches"] = True
        report["validations"]["csv_class_counts_match"] = True
        report["exported_at"] = exported_at
        manifest = {
            "dataset_name": settings.table, "source_schema": settings.schema,
            "source_table": settings.table, "row_count": stats["row_count"],
            "column_count": len(EXPORT_COLUMNS), "exported_at": exported_at,
            "feature_versions": [row["feature_version"] for row in stats["feature_version_distribution"]],
            "sha256": verification["sha256"], "size_bytes": verification["size_bytes"],
            "columns": EXPORT_COLUMNS, "encoding": "UTF-8", "null_representation": "unquoted empty field",
            "timezone": "UTC", "timestamp_format": "ISO 8601", "order_by": ["appointment_at", "appointment_id"],
            "exporter_version": "0.1.0", "python_version": platform.python_version(),
            "psycopg_version": psycopg.__version__, "strategy": "COPY TO STDOUT",
            "batch_size": size, "batch_size_unit": "local_buffer_bytes",
            "transaction": report["transaction"],
        }
        staged_manifest = staging / "manifest.json"
        staged_quality = staging / "quality.json"
        for path, value in ((staged_manifest, manifest), (staged_quality, report)):
            with path.open("w", encoding="utf-8") as destination:
                destination.write(report_json(value))
                destination.flush()
                os.fsync(destination.fileno())
        publish_artifacts([
            (staged_manifest, manifest_path), (staged_quality, quality_path),
            (staged_csv, csv_path),
        ], staging)
    logger.info("event=export_complete rows=%d bytes=%d attended=%d no_show=%d",
                stats["row_count"], verification["size_bytes"], stats["attended_count"], stats["no_show_count"])
    return ExportResult(csv_path, manifest_path, quality_path, stats["row_count"],
                        verification["size_bytes"], stats["attended_count"], stats["no_show_count"])
