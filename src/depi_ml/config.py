"""Configuración externa; las credenciales nunca forman parte del repr."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from collections.abc import Mapping

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Configuración incompleta o inválida."""


def positive_integer(value: str, name: str, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise ConfigurationError(f"{name} debe ser un entero positivo.") from None
    if number <= 0 or (maximum is not None and number > maximum):
        raise ConfigurationError(f"{name} está fuera del rango permitido.")
    return number


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)
    schema: str = "analytics"
    table: str = "appointment_training_dataset_v1"
    export_dir: Path = Path("data/exports")
    batch_size: int = 10000

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None,
        env_file: Path | None = Path(".env"),
    ) -> "Settings":
        values = dict(dotenv_values(env_file)) if env_file is not None else {}
        values.update(os.environ if environ is None else environ)

        def required(name: str, default: str | None = None) -> str:
            value = values.get(name, default)
            if value is None or not str(value).strip():
                raise ConfigurationError(f"Falta configurar {name}.")
            return str(value)

        password = required("POSTGRES_PASSWORD")
        if password == "CHANGE_ME":
            raise ConfigurationError("Reemplaza POSTGRES_PASSWORD en tu .env local.")
        return cls(
            host=required("POSTGRES_HOST", "127.0.0.1"),
            port=positive_integer(required("POSTGRES_PORT", "5432"), "POSTGRES_PORT", 65535),
            database=required("POSTGRES_DB", "depi_noshow"),
            user=required("POSTGRES_USER", "depi_admin"),
            password=password,
            schema=required("DATASET_SCHEMA", "analytics"),
            table=required("DATASET_TABLE", "appointment_training_dataset_v1"),
            export_dir=Path(required("DATASET_EXPORT_DIR", "data/exports")),
            batch_size=positive_integer(required("DATASET_BATCH_SIZE", "10000"), "DATASET_BATCH_SIZE"),
        )
