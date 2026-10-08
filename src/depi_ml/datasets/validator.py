"""Validación mediante agregados: no se cargan las citas en memoria."""

from datetime import datetime, timezone
import json

from psycopg import sql

from depi_ml.config import Settings
from depi_ml.datasets.schema import COLUMNS, COMPATIBLE_TYPES, EXPORT_COLUMNS


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str], report: dict | None = None):
        super().__init__(" ".join(errors))
        self.errors = errors
        self.report = report


def json_default(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    raise TypeError("Tipo no serializable en el reporte.")


def report_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False,
                      default=json_default) + "\n"


def validate_columns(actual: dict[str, str]) -> list[str]:
    errors = []
    for column in COLUMNS:
        if column.name not in actual:
            errors.append(f"Falta la columna requerida {column.name}.")
        elif actual[column.name] not in COMPATIBLE_TYPES[column.pg_type]:
            # No registrar valores arbitrarios obtenidos de la base.
            errors.append(f"Tipo incompatible en {column.name}; esperado {column.pg_type}.")
    return errors


def validate_statistics(stats: dict) -> list[str]:
    checks = [
        (stats["row_count"] == 0, "La tabla está vacía."),
        (stats["invalid_target_count"] > 0, "target contiene valores distintos de 0 y 1 o nulos."),
        (stats["duplicate_appointment_ids"] > 0, "Existen identificadores de cita duplicados."),
        (stats["null_counts"]["appointment_id"] > 0, "Hay identificadores de cita nulos."),
        (stats["null_counts"]["client_id"] > 0, "Hay identificadores de cliente nulos."),
        (stats["missing_prediction_dates"] > 0, "Faltan fechas de predicción o de cita."),
        (stats["invalid_temporal_count"] > 0, "prediction_at debe ser anterior a appointment_at."),
        (stats["nonfinite_timestamp_count"] > 0, "Hay timestamps infinitos en el contrato."),
    ]
    return [message for failed, message in checks if failed]


def inspect_dataset(connection, settings: Settings) -> dict:
    schema_row = connection.execute(
        "SELECT oid FROM pg_namespace WHERE nspname = %s", (settings.schema,),
    ).fetchone()
    if schema_row is None:
        raise DatasetValidationError(["No existe el schema configurado."])
    usage = connection.execute(
        "SELECT has_schema_privilege(%s::oid, 'USAGE') AS allowed", (schema_row["oid"],),
    ).fetchone()
    if not usage["allowed"]:
        raise DatasetValidationError(["Sin permiso USAGE sobre el schema configurado."])
    relation = connection.execute(
        "SELECT c.oid FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relname = %s AND c.relkind IN ('r', 'p', 'v', 'm', 'f')",
        (settings.schema, settings.table),
    ).fetchone()
    if relation is None:
        raise DatasetValidationError(["No existe la tabla o vista configurada."])
    column_rows = connection.execute(
        "SELECT a.attname, t.typname, "
        "has_column_privilege(a.attrelid, a.attnum, 'SELECT') AS readable "
        "FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid "
        "WHERE a.attrelid = %s AND a.attnum > 0 AND NOT a.attisdropped",
        (relation["oid"],),
    ).fetchall()
    actual = {row["attname"]: row["typname"] for row in column_rows}
    errors = validate_columns(actual)
    if any(not row["readable"] for row in column_rows if row["attname"] in EXPORT_COLUMNS):
        errors.append("Sin permiso SELECT sobre todas las columnas del contrato.")
    if errors:
        raise DatasetValidationError(errors)

    table = sql.Identifier(settings.schema, settings.table)
    aggregates = sql.SQL(
        "count(*) AS row_count, "
        "count(*) FILTER (WHERE target = 0) AS attended_count, "
        "count(*) FILTER (WHERE target = 1) AS no_show_count, "
        "count(*) FILTER (WHERE target IS NULL OR target NOT IN (0, 1)) AS invalid_target_count, "
        "count(appointment_id) - count(DISTINCT appointment_id) AS duplicate_appointment_ids, "
        "count(*) FILTER (WHERE prediction_at IS NULL OR appointment_at IS NULL) AS missing_prediction_dates, "
        "count(*) FILTER (WHERE prediction_at >= appointment_at) AS invalid_temporal_count, "
        "count(*) FILTER (WHERE NOT isfinite(prediction_at) OR NOT isfinite(appointment_at) "
        "OR NOT isfinite(built_at)) AS nonfinite_timestamp_count, "
        "min(appointment_at) FILTER (WHERE isfinite(appointment_at)) AS appointment_at_min, "
        "max(appointment_at) FILTER (WHERE isfinite(appointment_at)) AS appointment_at_max, "
        "count(DISTINCT client_id) AS unique_clients, "
        "count(DISTINCT clinic_id) AS distinct_clinics, "
        "count(*) FILTER (WHERE no_show_uid_minus_one IS TRUE) AS no_show_uid_minus_one_count, "
        "count(*) FILTER (WHERE age_at_booking < 0) AS negative_age_count, "
        "count(*) FILTER (WHERE age_at_booking > 120) AS age_above_120_count, "
        "count(*) FILTER (WHERE booking_lead_days < 0) AS negative_booking_lead_count, "
        "count(*) FILTER (WHERE booking_lead_days IN "
        "('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8)) AS nonfinite_booking_lead_count"
    )
    nulls = sql.SQL(", ").join(
        sql.SQL("count(*) FILTER (WHERE {} IS NULL) AS {}").format(
            sql.Identifier(name), sql.Identifier("null_" + name),
        ) for name in EXPORT_COLUMNS
    )
    stats = dict(connection.execute(
        sql.SQL("SELECT {}, {} FROM {}").format(aggregates, nulls, table),
    ).fetchone())
    stats["null_counts"] = {name: stats.pop("null_" + name) for name in EXPORT_COLUMNS}
    stats["no_show_percentage"] = (
        stats["no_show_count"] * 100 / stats["row_count"] if stats["row_count"] else 0.0
    )
    stats["feature_version_distribution"] = [dict(row) for row in connection.execute(
        sql.SQL("SELECT feature_version, count(*) AS row_count FROM {} "
                "GROUP BY feature_version ORDER BY feature_version NULLS LAST").format(table),
    )]
    errors = validate_statistics(stats)
    warnings = []
    for key in ("negative_age_count", "age_above_120_count", "negative_booking_lead_count",
                "nonfinite_booking_lead_count"):
        if stats[key]:
            warnings.append({"check": key, "count": stats[key]})
    report = {
        "source_schema": settings.schema, "source_table": settings.table,
        "checked_at": datetime.now(timezone.utc),
        "status": "failed" if errors else "passed",
        "validations": {
            "database_accessible": True, "schema_exists": True, "table_exists": True,
            "required_columns_present": True, "types_compatible": True,
            "read_permissions": True, "nonempty": stats["row_count"] > 0,
            "binary_target_without_nulls": stats["invalid_target_count"] == 0,
            "unique_appointment_ids": stats["duplicate_appointment_ids"] == 0,
            "identifiers_not_null": all(stats["null_counts"][n] == 0
                                        for n in ("appointment_id", "client_id")),
            "prediction_before_appointment": stats["invalid_temporal_count"] == 0,
            "prediction_dates_present": stats["missing_prediction_dates"] == 0,
            "finite_timestamps": stats["nonfinite_timestamp_count"] == 0,
        },
        "statistics": stats, "errors": errors, "warnings": warnings,
        "warning_rules": {
            "age": "Edad negativa o mayor a 120: alerta descriptiva, sin exclusión.",
            "booking_lead_days": "Anticipación negativa o no finita: alerta, sin modificación.",
        },
        "column_count": len(EXPORT_COLUMNS), "columns": EXPORT_COLUMNS,
        "column_types": {name: actual[name] for name in EXPORT_COLUMNS},
        "extra_source_columns": sorted(set(actual) - set(EXPORT_COLUMNS)),
        "transaction": {"isolation": "REPEATABLE READ", "read_only": True},
    }
    if errors:
        raise DatasetValidationError(errors, report)
    return report
