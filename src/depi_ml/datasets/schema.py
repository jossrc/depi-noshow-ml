"""Única fuente del orden y roles de las 27 columnas del contrato v1."""

from dataclasses import dataclass


PREDICTOR_COLUMNS = [
    "age_at_booking", "client_sex", "booking_lead_days", "appointment_month",
    "appointment_weekday", "appointment_hour", "duration_minutes", "is_fwa",
    "clinic_id", "scheduled_service_lines", "distinct_body_areas",
    "single_body_area_id", "has_medical_evaluation", "has_type4_service",
    "previous_attended", "previous_no_show", "previous_no_show_rate",
    "days_since_previous_attended", "days_since_previous_no_show",
]
IDENTIFIER_COLUMNS = ["appointment_id", "client_id"]
DATE_COLUMNS = ["prediction_at", "appointment_at"]
TARGET_COLUMN = "target"
AUDIT_COLUMNS = ["no_show_uid_minus_one"]
METADATA_COLUMNS = ["feature_version", "built_at"]


@dataclass(frozen=True)
class Column:
    name: str
    pg_type: str
    description: str
    role: str


_DEFINITIONS = [
    ("appointment_id", "int8", "Identificador de la cita"),
    ("client_id", "int8", "Identificador del cliente"),
    ("prediction_at", "timestamptz", "Momento de predicción/reserva"),
    ("appointment_at", "timestamptz", "Fecha y hora de la cita"),
    ("target", "int2", "0: atendida; 1: candidato no-show según FLOWww"),
    ("no_show_uid_minus_one", "bool", "Caso asociado al usuario de tracking -1"),
    ("age_at_booking", "int2", "Edad del cliente al reservar"),
    ("client_sex", "varchar", "Sexo registrado; sin recodificación"),
    ("booking_lead_days", "float8", "Anticipación de la reserva en días"),
    ("appointment_month", "int2", "Mes de la cita"),
    ("appointment_weekday", "int2", "Día de semana; convención definida por el SQL fuente"),
    ("appointment_hour", "int2", "Hora de la cita"),
    ("duration_minutes", "int4", "Duración programada en minutos"),
    ("is_fwa", "bool", "Indicador operativo FWA definido por FLOWww"),
    ("clinic_id", "int4", "Identificador de clínica"),
    ("scheduled_service_lines", "int4", "Número de líneas de servicio programadas"),
    ("distinct_body_areas", "int4", "Número de áreas corporales distintas"),
    ("single_body_area_id", "int8", "Área corporal cuando hay una única área"),
    ("has_medical_evaluation", "bool", "Incluye evaluación médica"),
    ("has_type4_service", "bool", "Incluye servicio de tipo 4"),
    ("previous_attended", "int4", "Asistencias anteriores conocidas al predecir"),
    ("previous_no_show", "int4", "No-shows anteriores conocidos al predecir"),
    ("previous_no_show_rate", "float8", "Proporción histórica de no-shows"),
    ("days_since_previous_attended", "float8", "Días desde la asistencia anterior"),
    ("days_since_previous_no_show", "float8", "Días desde el no-show anterior"),
    ("feature_version", "text", "Versión de construcción de características"),
    ("built_at", "timestamptz", "Momento de construcción del registro analítico"),
]


def _role(name: str) -> str:
    if name in PREDICTOR_COLUMNS:
        return "predictor"
    if name in IDENTIFIER_COLUMNS:
        return "identificador"
    if name in DATE_COLUMNS:
        return "fecha de auditoría"
    if name == TARGET_COLUMN:
        return "objetivo"
    if name in AUDIT_COLUMNS:
        return "auditoría"
    return "metadato"


COLUMNS = tuple(Column(name, pg_type, description, _role(name))
                for name, pg_type, description in _DEFINITIONS)
EXPORT_COLUMNS = [column.name for column in COLUMNS]
TIMESTAMP_COLUMNS = [column.name for column in COLUMNS if column.pg_type == "timestamptz"]
COMPATIBLE_TYPES = {
    "int2": {"int2", "int4", "int8"},
    "int4": {"int2", "int4", "int8"},
    "int8": {"int2", "int4", "int8"},
    "float8": {"float4", "float8"},
    "varchar": {"varchar", "text", "bpchar"},
    "text": {"varchar", "text", "bpchar"},
    "bool": {"bool"},
    "timestamptz": {"timestamptz"},
}
