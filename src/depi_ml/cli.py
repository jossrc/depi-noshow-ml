"""Comandos de exportación y experimentos locales; dependencias ML bajo demanda."""

import argparse
import csv
import logging
from pathlib import Path

import psycopg

from depi_ml.config import ConfigurationError, Settings, positive_integer
from depi_ml.db.connection import read_only_connection
from depi_ml.datasets.exporter import ExportError, export_dataset
from depi_ml.datasets.validator import DatasetValidationError, inspect_dataset, report_json
from depi_ml.errors import Phase2Error


def _batch_size(value: str) -> int:
    try:
        return positive_integer(value, "--batch-size")
    except ConfigurationError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _label_temporal_status(summary, logger):
    """Advertencias operativas no cambian el código de salida; errores críticos sí."""
    critical = summary["critical_temporal_errors"]["rows"]
    operational = summary["operational_warnings"]["rows"]
    print(f"Anomalías temporales críticas: {critical}; advertencias operativas: {operational}.")
    if operational:
        logger.warning("event=label_operational_warning count=%d message=Asistencias conservadas; revisión humana pendiente, causas individuales no demostradas.", operational)
    if critical:
        logger.error("event=label_dates_inconsistent count=%d message=Consulta el reporte agregado; entrenamiento bloqueado.", critical)
        return 1
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Exportar, auditar y experimentar con el dataset histórico DEPI.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-dataset", help="Validar estructura y estadísticas sin exportar.")
    export = commands.add_parser("export-dataset", help="Exportar todas las filas y reportes.")
    export.add_argument("--output", type=Path, help="Ruta CSV de salida.")
    export.add_argument("--batch-size", type=_batch_size,
                        help="Bytes del buffer local COPY; también intervalo de filas del log de verificación.")
    analyze = commands.add_parser("analyze-dataset", help="Auditar CSV y manifest locales sin PostgreSQL.")
    train = commands.add_parser("train", help="Entrenar con revisión acreditada y reservar prueba temporal.")
    evaluate = commands.add_parser("evaluate", help="Evaluar la prueba reservada de un experimento local propio.")
    label_export = commands.add_parser("export-label-availability", help="Exportar marcas por cita en PostgreSQL READ ONLY sin sobrescribir el dataset.")
    label_validate = commands.add_parser("validate-label-availability", help="Validar el auxiliar y previsualizar cortes sin entrenar ni conectar a PostgreSQL.")
    optimize = commands.add_parser("optimize-exploratory", help="Una ronda fija para full/without_catalog, solo validación y sin prueba.")
    for command in [analyze, train, evaluate, label_export, label_validate, optimize]:
        command.add_argument("--csv", type=Path, default=Path("data/exports/training_dataset_v1.csv"))
        command.add_argument("--manifest", type=Path, help="Por defecto, <nombre_csv>_manifest.json.")
        command.add_argument("--output", type=Path, required=True,
                             help="CSV auxiliar nuevo." if command is label_export else "Carpeta nueva fuera de data/exports.")
    for command in [train, evaluate, label_validate, optimize]:
        command.add_argument("--labels", type=Path, required=True, help="CSV auxiliar de label_recorded_at.")
        command.add_argument("--labels-manifest", type=Path, help="Por defecto, <nombre_auxiliar>_manifest.json.")
    for name in ["validation-start", "test-start", "test-end"]:
        label_validate.add_argument(f"--{name}", help="Opcional: los tres cortes para una vista de particiones sin entrenamiento.")
    analyze.add_argument("--timezone", default="America/Lima")
    analyze.add_argument("--min-group", type=_batch_size, default=10)
    train.add_argument("--review", type=Path, help="JSON de revisión con evidencia, ligado al hash del CSV.")
    optimize.add_argument("--reference", type=Path, required=True, help="Experimento exploratorio original terminado; no se sobrescribe.")
    optimize.add_argument("--review", type=Path, help="Plantilla metodológica original; se conserva sin verificar estados.")
    train.add_argument("--exploratory", action="store_true",
                       help="Permitir controles metodológicos pendientes; solo entrenamiento/validación, EXPLORATORY_NOT_VALIDATED.")
    for name in ["validation-start", "test-start", "test-end"]:
        train.add_argument(f"--{name}", required=True, help="ISO 8601 con zona explícita; corte por fecha de reserva.")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threshold", type=float, default=.5, help="Umbral diagnóstico fijado antes de evaluar prueba.")
    train.add_argument("--min-group", type=_batch_size, default=30)
    train.add_argument("--importance-samples", type=_batch_size, default=2000)
    evaluate.add_argument("--experiment", type=Path, required=True)
    evaluate.add_argument("--shap", action="store_true", help="Agregar importancias SHAP agregadas; requiere .[explain].")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s level=%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    try:
        if args.command == "export-label-availability":
            try:
                from depi_ml.datasets.label_exporter import export_label_availability
            except ImportError:
                raise Phase2Error("Instala '.[ml]' para validar el CSV original y exportar el auxiliar.") from None
            labels, manifest, quality, summary = export_label_availability(Settings.from_env(), args.csv, args.manifest, args.output)
            print(f"Auxiliar local: {labels}\nManifest: {manifest}\nCalidad: {quality}\n"
                  f"Registros: {summary['rows']}; fechas ausentes: {summary['missing_label_recorded_at']}.\n"
                  "label_recorded_at es una marca registrada; disponibilidad real pendiente de validación humana.")
            return _label_temporal_status(summary, logger)
        elif args.command in {"analyze-dataset", "train", "evaluate", "validate-label-availability", "optimize-exploratory"}:
            try:
                if args.command == "analyze-dataset":
                    from depi_ml.analysis.reports import analyze
                    output = analyze(args.csv, args.manifest, args.output, args.timezone, args.min_group)
                    print(f"Análisis local: {output}\nRevisa audit.json y methodology.md; entrenamiento pendiente de revisión metodológica.")
                elif args.command == "train":
                    from depi_ml.training.experiments import train
                    output = train(args.csv, args.manifest, args.output, args.review, args.validation_start,
                                   args.test_start, args.test_end, args.seed, args.threshold, args.min_group, args.importance_samples,
                                   labels_path=args.labels, labels_manifest_path=args.labels_manifest,
                                   exploratory=args.exploratory)
                    if args.exploratory:
                        print(f"Experimento local: {output}\nEXPLORATORY_NOT_VALIDATED: validación exploratoria completada; "
                              "prueba reservada y bloqueada para evaluate. Revisión metodológica pendiente.")
                    else:
                        print(f"Experimento local: {output}\nValidación completada; prueba reservada para evaluate. Modelos experimentales.")
                elif args.command == "evaluate":
                    from depi_ml.evaluation.runner import evaluate
                    output = evaluate(args.csv, args.manifest, args.experiment, args.output, args.shap,
                                      labels_path=args.labels, labels_manifest_path=args.labels_manifest)
                    print(f"Evaluación de prueba: {output}\nResultados experimentales; no constituyen aprobación de producción.")
                elif args.command == "optimize-exploratory":
                    from depi_ml.training.optimization import optimize
                    output = optimize(args.csv, args.manifest, args.labels, args.labels_manifest, args.reference,
                                      args.output, args.review)
                    print(f"Optimización local: {output}\nEXPLORATORY_NOT_VALIDATED: una ronda completada; "
                          "umbral 0.5, semilla 42, prueba reservada sin evaluar. No se seleccionó modelo definitivo.")
                else:
                    from depi_ml.analysis.label_report import validate_labels
                    output, report = validate_labels(args.csv, args.manifest, args.labels, args.labels_manifest, args.output,
                                                     args.validation_start, args.test_start, args.test_end)
                    print(f"Validación local: {output}\nRegistros: {report['validation']['rows']}; "
                          f"fechas ausentes: {report['validation']['missing_label_recorded_at']}.\n"
                          "No se entrenaron modelos ni se verificó automáticamente la revisión metodológica.")
                    return _label_temporal_status(report["validation"], logger)
            except ImportError:
                raise Phase2Error("Faltan dependencias de Fase 2: instala '.[ml]' y, para SHAP, '.[explain]'.") from None
        elif args.command == "inspect-dataset":
            settings = Settings.from_env()
            logger.info("event=inspection_start")
            with read_only_connection(settings) as connection:
                report = inspect_dataset(connection, settings)
            print(report_json(report), end="")
            logger.info("event=inspection_complete rows=%d", report["statistics"]["row_count"])
        else:
            settings = Settings.from_env()
            result = export_dataset(settings, args.output, args.batch_size)
            print(f"Exportación completada: {result.row_count} filas, {result.size_bytes} bytes.\n"
                  f"Asistencias: {result.attended_count}; candidatos no-show: {result.no_show_count}.\n"
                  f"CSV: {result.csv_path}\nManifest: {result.manifest_path}\nCalidad: {result.quality_path}")
    except (ConfigurationError, DatasetValidationError, ExportError, Phase2Error) as error:
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
