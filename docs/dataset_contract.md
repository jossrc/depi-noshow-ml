# Contrato del dataset v1

Fuente oficial: `analytics.appointment_training_dataset_v1`. El paquete `depi_ml.datasets.schema` centraliza todas las columnas y su orden. La extracción no consulta `raw_flow`, no recalcula features y no ejecuta ETL. Se exportan 27 columnas: 19 predictoras y 8 campos de objetivo, trazabilidad, auditoría y metadatos.

| Columna | Tipo PostgreSQL | Rol | Significado |
| --- | --- | --- | --- |
| appointment_id | BIGINT | Identificador | Cita real; trazabilidad y detección de duplicados |
| client_id | BIGINT | Identificador | Cliente; agrupación y auditoría |
| prediction_at | TIMESTAMPTZ | Fecha de auditoría | Momento de predicción/reserva; debe preceder a la cita |
| appointment_at | TIMESTAMPTZ | Fecha de auditoría | Fecha de cita; orden y futura evaluación temporal |
| target | SMALLINT | Objetivo | 0: atendida; 1: candidato no-show reconstruido con reglas FLOWww |
| no_show_uid_minus_one | BOOLEAN | Auditoría | Asociado a tracking UID -1; conservar para futuros experimentos |
| age_at_booking | SMALLINT | Predictor | Edad al reservar |
| client_sex | VARCHAR(1) | Predictor | Sexo registrado sin recodificar |
| booking_lead_days | DOUBLE PRECISION | Predictor | Anticipación de la reserva en días |
| appointment_month | SMALLINT | Predictor | Mes de la cita |
| appointment_weekday | SMALLINT | Predictor | Día de semana según la convención del SQL fuente, aún por confirmar científicamente |
| appointment_hour | SMALLINT | Predictor | Hora de cita |
| duration_minutes | INTEGER | Predictor | Duración programada en minutos |
| is_fwa | BOOLEAN | Predictor | Indicador operativo FWA; semántica definida por FLOWww |
| clinic_id | INTEGER | Predictor | Clínica de la cita |
| scheduled_service_lines | INTEGER | Predictor | Número de líneas de servicio programadas |
| distinct_body_areas | INTEGER | Predictor | Áreas corporales diferentes |
| single_body_area_id | BIGINT | Predictor | Identificador cuando corresponde una sola área |
| has_medical_evaluation | BOOLEAN | Predictor | Incluye evaluación médica |
| has_type4_service | BOOLEAN | Predictor | Incluye servicio de tipo 4 |
| previous_attended | INTEGER | Predictor | Asistencias anteriores conocidas al predecir |
| previous_no_show | INTEGER | Predictor | No-shows anteriores conocidos al predecir |
| previous_no_show_rate | DOUBLE PRECISION | Predictor | Tasa histórica de no-shows |
| days_since_previous_attended | DOUBLE PRECISION | Predictor | Días desde asistencia previa |
| days_since_previous_no_show | DOUBLE PRECISION | Predictor | Días desde no-show previo |
| feature_version | TEXT | Metadato | Versión de preparación SQL de las características |
| built_at | TIMESTAMPTZ | Metadato | Fecha de construcción del registro analítico |

Los identificadores, `target`, `no_show_uid_minus_one`, las fechas y los metadatos **no forman parte de las características predictoras**. En Fase 2 `clinic_id`, `single_body_area_id`, sexo, booleanos, mes, día y hora se codifican como categorías dentro de un pipeline ajustado exclusivamente con entrenamiento. El contrato de roles permanece en `datasets/schema.py`; ningún módulo redefine una lista alternativa de 19 predictoras.

## Compatibilidad y representación

Se aceptan `int2/int4/int8` para enteros, `float4/float8` para flotantes y `varchar/text/bpchar` para texto. Estos tipos se exportan según su representación original, sin cast numérico. Booleanos requieren `bool` y fechas requieren `timestamptz`; se rechazan tipos de dominio o tipos distintos, incluso si PostgreSQL permite conversiones implícitas. La compatibilidad de texto no certifica la longitud ni las categorías de sexo. Las columnas adicionales del origen se anotan en el reporte y se omiten del CSV: el contrato v1 permanece explícito y estable.

El CSV usa UTF-8, separador coma, encabezados en el orden de esta tabla y reglas de comillas de PostgreSQL. Nulos: campos vacíos sin comillas. Texto vacío: `""`. Booleanos: `t`/`f`. Timestamps: UTC, `T` entre fecha y hora y offset `+00`, con precisión original disponible. No se imputan ni eliminan nulos y no se modifican categorías. El hash SHA-256 representa los bytes exactos del archivo.

## Criterios técnicos

Errores críticos: acceso o permisos insuficientes; fuente inexistente; columnas ausentes o tipos incompatibles; cero registros; target nulo o fuera de `{0,1}`; appointment_id duplicado/nulo o client_id nulo; prediction_at/appointment_at ausentes; prediction_at mayor o igual a appointment_at; timestamps infinitos. Detienen la exportación sin corregir datos.

Las predictoras nulas se conservan y cuentan por columna. Se notifican edades menores a 0 o mayores a 120 como valores para revisión descriptiva, sin establecer criterios de inclusión. También se notifican anticipaciones negativas o no finitas. No se aplican umbrales de exclusión para género, clínica, servicio o anticipación elevada. Los extremos de edad no implican automáticamente que el registro sea incorrecto.

La validación, estadísticas y COPY usan el mismo snapshot `REPEATABLE READ READ ONLY`. Se contrastan encabezados, cantidad de columnas, total de registros lógicos y distribución de clases del CSV con ese snapshot. Los reportes incluyen versión de features, nulos, cantidades de tracking -1, extremos de fechas, clientes/clínicas únicos, advertencias y hash; no incluyen filas individuales.

## Limitaciones conocidas para la tesis

- La etiqueta no-show se reconstruyó con reglas operativas FLOWww; `target=1` significa candidato no-show, pendiente de validación metodológica.
- El tracking UID -1 permanece identificado; futuros experimentos podrán comparar su inclusión y exclusión.
- Las cancelaciones previas no constituyen asistencias y los desplazamientos históricos no deben duplicar citas reales. La extracción no verifica ni reconstruye estas reglas fuente.
- Los saldos actuales de bonos no son adecuados para predecir retrospectivamente citas históricas.
- `client_history_current` no es seguro para entrenamiento retrospectivo; no se consulta para esta exportación.
- Ninguna feature debería depender de información posterior a `prediction_at`. La comprobación prediction_at < appointment_at es necesaria pero no demuestra ausencia de fuga temporal en todas las variables.
- Campos del catálogo actual pueden haber cambiado respecto a la reserva histórica. Su vigencia retrospectiva aún requiere revisión.
- Algunas reglas temporales, la convención de día de semana y la semántica operativa de ciertos campos siguen pendientes de validación científica.

El exportador conserva los datos analíticos existentes y documenta estas limitaciones. No intenta resolverlas mediante filtros, reconstrucción del historial o cambios de datos. La Fase 2 divide por `prediction_at` en entrenamiento/validación/prueba y purga etiquetas que no estarían maduras antes del siguiente corte, usando una cota de demora acreditada externamente. Esta política no certifica retrospectivamente la disponibilidad real por fila: el CSV carece de `known_at`/`outcome_resolved_at`. Véase [el procedimiento metodológico](phase2.md).
