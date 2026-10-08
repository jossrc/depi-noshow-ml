"""Interfaz de comandos de la fase 1."""

import argparse
import csv
import logging
from pathlib import Path

import psycopg

from depi_ml.config import ConfigurationError, Settings, positive_integer
from depi_ml.db.connection import read_only_connection
from depi_ml.datasets.exporter import ExportError, export_dataset
from depi_ml.datasets.validator import DatasetValidationError, inspect_dataset, report_json


def _batch_size(value: str) -> int:
    try:
        return positive_integer(value, "--batch-size")
    except ConfigurationError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Validar y exportar el dataset histórico DEPI.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-dataset", help="Validar estructura y estadísticas sin exportar.")
    export = commands.add_parser("export-dataset", help="Exportar todas las filas y reportes.")
    export.add_argument("--output", type=Path, help="Ruta CSV de salida.")
    export.add_argument("--batch-size", type=_batch_size,
                        help="Bytes del buffer local COPY; también intervalo de filas del log de verificación.")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s level=%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    try:
        settings = Settings.from_env()
        if args.command == "inspect-dataset":
            logger.info("event=inspection_start")
            with read_only_connection(settings) as connection:
                report = inspect_dataset(connection, settings)
            print(report_json(report), end="")
            logger.info("event=inspection_complete rows=%d", report["statistics"]["row_count"])
        else:
            result = export_dataset(settings, args.output, args.batch_size)
            print(f"Exportación completada: {result.row_count} filas, {result.size_bytes} bytes.\n"
                  f"Asistencias: {result.attended_count}; candidatos no-show: {result.no_show_count}.\n"
                  f"CSV: {result.csv_path}\nManifest: {result.manifest_path}\nCalidad: {result.quality_path}")
    except (ConfigurationError, DatasetValidationError, ExportError) as error:
        logger.error("event=validation_failed message=%s", str(error))
        return 1
    except psycopg.Error as error:
        # Mensajes PostgreSQL pueden incluir datos personales de filas o credenciales.
        code = error.sqlstate
        if code == "42501":
            message = "Sin permisos de lectura; verifica USAGE del schema y SELECT de las columnas."
        elif code == "28P01":
            message = "Autenticación rechazada; verifica el usuario y la contraseña local."
        elif code == "3D000":
            message = "La base configurada no existe."
        else:
            message = "Falló el acceso a PostgreSQL; verifica host, puerto publicado, base, credenciales y disponibilidad."
        logger.error("event=database_failed message=%s", message)
        return 1
    except (OSError, UnicodeError, ValueError, csv.Error):
        logger.error("event=export_failed message=No se pudo generar o verificar la salida; revisa permisos, espacio y formato.")
        return 1
    except KeyboardInterrupt:
        logger.error("event=cancelled message=Operación cancelada.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
