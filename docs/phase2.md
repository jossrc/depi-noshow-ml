# Fase 2: análisis y experimentos locales

Todo permanece en `depi-noshow-ml`. No se reconstruye la Fase 1 ni se implementa FastAPI. `analyze-dataset`, `validate-label-availability`, `train` y `evaluate` trabajan con archivos locales y no leen configuración de conexión. `export-label-availability` obtiene un auxiliar nuevo usando la transacción PostgreSQL `REPEATABLE READ READ ONLY` existente. Ningún comando nuevo modifica tablas, Docker, el CSV/manifest original o las 19 variables predictoras.

## Ejecutar primero la auditoría

Desde la raíz, utilizando el entorno existente:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,ml,explain]'
# Solo en macOS si falta OpenMP:
brew install libomp
depi-ml analyze-dataset \
  --csv data/exports/training_dataset_v1.csv \
  --manifest data/exports/training_dataset_v1_manifest.json \
  --output reports/phase2-review-01
python -m pytest -q
```

No copies `.env.example` sobre un `.env` existente. No necesitas iniciar/recrear Docker ni exportar de nuevo. `--output` exige una carpeta nueva y fuera de la carpeta del CSV fuente para evitar sobrescrituras. La zona descriptiva predeterminada es `America/Lima`; se puede cambiar con `--timezone`. Los cortes de entrenamiento se normalizan a UTC. Un error termina con código 1, argumentos inválidos con 2, cancelación con 130. Sin CSV/manifest no se generan estadísticas reales.

La lectura analítica utiliza memoria proporcional al CSV, a diferencia del exportador streaming. Los nulos se conservan; el análisis no imputa, elimina ni recorta registros. Las representaciones vacías de features del CSV se interpretan como ausentes durante análisis/entrenamiento, sin cambiar bytes del archivo fuente. Valores extremos por IQR son señales descriptivas, no criterios de exclusión.

## Extraer marcas registradas por cita

La demora global `appointment_at + label_delay_hours` se retiró. Los cierres administrativos tardíos hacen que una cota global no describa todas las citas. El auxiliar tiene cinco columnas: `appointment_id`, `target`, `outcome`, `label_recorded_at`, `label_source`. Los identificadores se usan para correspondencia y permanecen exclusivamente en el archivo local ignorado; los reportes solo muestran agregados.

| Target | Outcome esperado | Marca registrada |
| --- | --- | --- |
| 0 | `ATTENDED_COMPLETED` | `raw_flow.laser_gen.LaserGCloseDate AT TIME ZONE 'America/Lima'` |
| 1 | `NO_SHOW` | `analytics.appointment_outcomes.outcome_resolved_at` |

Se une la tabla de entrenamiento con `appointment_outcomes` por `appointment_id` y, para asistencias, con `laser_gen.LaserGID` mediante `appointment_outcomes.diary_laser_id`. No se toma el resultado de una cita/cliente/clínica diferente. Se exige `LaserGCloseDate` de tipo `timestamp` sin zona y `outcome_resolved_at` de tipo `timestamptz`; un cambio de tipos bloquea la extracción. La conversión interpreta el cierre en Lima y escribe UTC, siguiendo la operación [AT TIME ZONE de PostgreSQL](https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-ZONECONVERT).

Con el `.env` existente y PostgreSQL ya accesible, ejecutar **una sola vez por destino nuevo**:

```bash
depi-ml export-label-availability \
  --csv data/exports/training_dataset_v1.csv \
  --manifest data/exports/training_dataset_v1_manifest.json \
  --output data/exports/training_dataset_v1_labels.csv
```

No inicies/recrees Docker ni ejecutes SQL de ETL para este ajuste. El comando solo usa SELECT y COPY TO STDOUT dentro de [REPEATABLE READ READ ONLY](https://www.postgresql.org/docs/current/sql-set-transaction.html). Primero reexporta temporalmente las 27 columnas con el COPY existente y verifica el **mismo hash exacto** del CSV original. Si la base cambió en filas, features, etiquetas o metadatos, se detiene sin publicar el auxiliar; incluso diferencias de representación que cambien bytes se rechazan conservadoramente. Los temporales permanecen locales y se eliminan al cerrar. Comprueba relaciones, tipos, unicidad, IDs, clases y fechas en esa misma instantánea.

Archivos locales nuevos:

```text
data/exports/training_dataset_v1_labels.csv
data/exports/training_dataset_v1_labels_manifest.json
data/exports/training_dataset_v1_labels_quality_report.json
```

Se verifica SHA-256 y tamaño del auxiliar. El manifest incluye los hashes del CSV y manifest originales, la fecha original de exportación, la instantánea de extracción, definiciones de fuentes y el carácter provisional de `label_recorded_at`. No contiene contraseñas. CSV y manifiestos generados están ignorados en Git. No se sobrescriben archivos originales ni auxiliares previos: utiliza otro nombre si necesitas repetir la extracción.

`label_recorded_at` representa la **marca temporal registrada**, todavía pendiente de revisión humana como aproximación de cuándo realmente se conoció la etiqueta. No equivale automáticamente a disponibilidad: pueden existir cierres retroactivos, ediciones, ingestas posteriores o diferencias entre reserva vigente y fecha histórica. No se coalescea con la fecha de cita ni se usa otra fuente para rellenar cierres ausentes.

Fechas faltantes se conservan como vacías y se reportan por clase. Una marca anterior a `appointment_at` o posterior a la instantánea de PostgreSQL es una inconsistencia crítica. El exportador conserva esas marcas sin corregirlas en un **auxiliar de auditoría bloqueado**, escribe el reporte y devuelve código 1; no permite usarlas en entrenamiento/evaluación. Identificadores duplicados, correspondencia/resultado incorrectos, fechas no interpretables/no finitas o cambio del CSV en PostgreSQL detienen la publicación. Una revisión humana no anula esos controles críticos.

## Validar el auxiliar sin PostgreSQL ni entrenamiento

```bash
depi-ml validate-label-availability \
  --csv data/exports/training_dataset_v1.csv \
  --labels data/exports/training_dataset_v1_labels.csv \
  --validation-start 2026-04-01T00:00:00-05:00 \
  --test-start 2026-07-01T00:00:00-05:00 \
  --test-end 2026-10-01T00:00:00-05:00 \
  --output reports/label-review-01
```

Sin los tres cortes, el comando valida solo archivos; si se indican cortes deben proporcionarse los tres. `--labels-manifest` permite especificar otro manifest; por defecto usa `<nombre_auxiliar>_manifest.json`. Produce `label_availability_report.json` y `methodology_review_template.json`. Ambos son agregados, sin citas individuales; **todos los controles de la plantilla siguen en `pending`**. Si hay fechas inconsistentes, conserva el reporte, devuelve código 1 y no genera una vista de particiones que pueda parecer válida. En casos técnicamente coherentes, previsualiza conteos sin exigir ambas clases, y advierte particiones insuficientes; `train` sí exige ambas clases.

La primera extracción real de este ajuste verificó las 213.873 citas y detectó 3.410 marcas anteriores a la cita, todas en asistencias, sin fechas ausentes. El cierre tardío máximo registrado fue aproximadamente 221 días. El auxiliar queda bloqueado para entrenamiento. Estos resultados corresponden al hash del CSV auditado; deben revisarse en el reporte local, no corregirse mediante un desplazamiento horario o recorte automático. El contexto de PROMPT3 declara verificado `known_at < prediction_at`; se conserva esa información, pero la plantilla no convierte automáticamente esa declaración en evidencia de revisión firmada.

## Reportes del análisis

| Archivo | Contenido |
| --- | --- |
| `completion.json` | Marca de análisis completo y hash del CSV |
| `audit.json`, `methodology.md` | Controles, bloqueos, advertencias, revisión SQL pendiente y alertas de tasas mensuales |
| `classes.csv`, `monthly.csv`, `clinics.csv`, `clinic_month.csv` | Asistencias, candidatos no-show, proporciones y evolución por clínica |
| `missing_values.csv`, `numeric_summary.csv`, `numeric_correlations.csv`, `outliers.json` | Nulos, distribución numérica, correlaciones y extremos sin datos individuales |
| `distribution_*.csv`, `history_groups.csv`, `recency_*.csv` | Categorías, tratamientos, historial y recencia |
| `uid_groups.csv`, `uid_month.csv`, `uid_feature_comparison.csv`, `uid_categories_*.csv` | UID -1 vs resto y comparación entre candidatos no-show de ambos grupos |
| `atypical_cohort.json`, `atypical_cohort_monthly.csv` si corresponde | Cohorte atípica solicitada, anonimizada y agregada |
| `hist_*.png`, `monthly_rate.png` | Distribuciones completas y evolución de tasas |
| `methodology_review_template.json` | Plantilla pendiente; **no habilita entrenamiento** |

Los grupos ordinarios con menos de 10 citas se omiten, incluyendo sus etiquetas de grupo. La cohorte atípica se describe por conteos de clases/UID incluso si es pequeña, pues fue solicitada expresamente; sus cuantiles y períodos se suprimen con menos de 10 citas. No se exportan identificadores de cita/cliente ni filas. Los IDs de clínica/área son categorías del contrato, no IDs personales. Los reportes agregados siguen siendo resultados locales: no se suben a Git automáticamente.

Las alertas mensuales usan un criterio descriptivo de diferencia >=5 puntos porcentuales y >5 errores estándar nominales con al menos 100 citas por mes. No son una prueba formal de concept drift: no corrigen multiplicidad ni dependencia por cliente; meses parciales, mezcla de clínicas y madurez de etiquetas necesitan revisión separada. No se altera el dataset por una alerta.

## Controles que bloquean entrenamiento

Los errores observables del CSV (por ejemplo duplicados, etiquetas inválidas, fechas imposibles, historial negativo, tasas/recencia incoherentes) bloquean entrenamiento aunque se presente una revisión. Una revisión no los anula. SHA-256 correcto y fechas ordenadas no acreditan ausencia de fuga de información.

El contrato original no incluye marcas de etiquetas ni snapshots de reservas o catálogo. El auxiliar incorpora marcas registradas por cita, pero su validez como aproximación de disponibilidad real sigue pendiente. Se requiere un JSON local de revisión vinculado al hash/versiones del dataset **y los hashes del auxiliar y su manifest**, con responsable, fecha con zona y evidencia documental. El programa valida estructura/declaraciones; la persona revisora debe acreditar la evidencia, no basta con editar estados para superar el control. No se acepta una demora global para sustituir las marcas por cita.

1. **history_point_in_time**: revisar el ETL completo que construyó las 19 features y los acumulados de `ml_history_cumulative_v1`. Acreditar `known_at < prediction_at`, conocimiento real de eventos y exclusión de fila propia/futuros; no usar `client_history_current` ni saldos actuales. Comparar conteos y recencias con eventos fuente. El CSV solo permite coherencia interna y comparación parcial con citas observadas, no prueba el join as-of.
2. **booking_snapshot**: comprobar si la reserva, clínica, horario, duración, sexo y servicios SCHEDULED representan lo que se conocía al reservar. Una fase SCHEDULED por sí sola no demuestra que el registro no se haya editado después. Revisar revisiones/reprogramaciones y evitar líneas PERFORMED/BILLED.
3. **label_availability**: validar `label_recorded_at` para ambas clases, la interpretación Lima del cierre manual, ediciones retroactivas y fecha de ingestión. Revisar `outcome_resolved_at`, eventos/cierres/tracking y `known_at`: una marca registrada no acredita cuándo fue visible al personal/sistema. Resolver anomalías de fechas antes de entrenar y documentar la exclusión de marcas ausentes/tardías por corte, que puede seleccionar diferencialmente asistencias y no-shows.
4. **catalog_snapshot**: acreditar catálogo as-of o invariancia histórica de áreas/tipos/evaluación médica. `product_catalog_resolution.updated_at` y un catálogo actual no bastan para demostrar confiabilidad retrospectiva.
5. **label_definition**: validar candidatos no-show, conflictos de tickets, tracking UID -1 y exclusiones de usuarios/clínicas. Revisar conversiones `timestamp`/`timestamptz` y reglas FLOWww con la zona real. `target=1` conserva la condición de candidato.
6. **selection_bias**: documentar que la evaluación está condicionada a citas elegibles y excluye cancelaciones anticipadas. Eso puede seleccionar usando un evento posterior a la reserva; no generalizar métricas a todas las reservas ni al futuro servicio sin redefinir la población.
7. **uncertain_semantics**: interpretar `is_fwa`, convención weekday y variables operativas. La sensibilidad sin `is_fwa` y sin catálogo no valida automáticamente su temporalidad.
8. **temporal_label_stability**: explicar cambios por mes/clínica, cobertura y reglas, distinguiendo cambios reales de tracking o etiquetas incompletas. No ocultar caídas abruptas antes de entrenar.

`bds/analytics.sql` contiene estructuras y una vista de clasificación; en la revisión inicial no está la consulta completa que pobló las 19 características. No se ejecuta ese archivo. Cualquier futura consulta a PostgreSQL debe realizarse usando una transacción `READ ONLY` como la que ya ofrece el proyecto.

Copia la plantilla a un archivo ignorado y complétala solo tras revisar evidencia real:

```bash
cp reports/label-review-01/methodology_review_template.json methodology_review.local.json
```

En `checks`, historia, reserva, disponibilidad, catálogo, definición de etiquetas y estabilidad temporal requieren `status: "verified"`; selección y semántica requieren `verified` o `acknowledged`, con evidencia y limitaciones en ambos casos. `reviewed_at` usa ISO 8601 con zona. Los campos `label_availability_sha256` y `label_availability_manifest_sha256` deben corresponder a los archivos revisados. La plantilla creada por `analyze-dataset`, sin auxiliar, deja esos hashes vacíos; utiliza la plantilla de `validate-label-availability`. El código conserva la revisión con los artefactos. No se verifican estados automáticamente por tener un archivo íntegro.

## Entrenar después de superar controles

Estos cortes son un **protocolo candidato**, que debe justificarse frente a las alertas y cobertura observadas antes de ejecutarlo; no resuelven el cambio de etiquetas:

```bash
depi-ml train \
  --csv data/exports/training_dataset_v1.csv \
  --labels data/exports/training_dataset_v1_labels.csv \
  --review methodology_review.local.json \
  --validation-start 2026-04-01T00:00:00-05:00 \
  --test-start 2026-07-01T00:00:00-05:00 \
  --test-end 2026-10-01T00:00:00-05:00 \
  --seed 42 \
  --output experiments/phase2-01
```

El comando falla sin auxiliar técnicamente coherente y revisión acreditada. Divide por **momento de reserva**, no por fecha de cita ni al azar, y mantiene el orden cronológico por `prediction_at`:

| Partición | Reservas | Marca registrada de etiqueta |
| --- | --- | --- |
| Entrenamiento | `prediction_at < validation_start` | `label_recorded_at < validation_start` |
| Validación | `validation_start <= prediction_at < test_start` | `label_recorded_at < test_start` |
| Prueba | `test_start <= prediction_at < test_end` | `label_recorded_at < exported_at` del **dataset original** |

Las marcas iguales al corte se excluyen. No se sustituye la fecha original por la fecha posterior del auxiliar. Las fechas ausentes se excluyen en memoria, sin imputación ni alteración del CSV. Los cierres días/meses después de la atención se evalúan por su marca real registrada y no por una demora global. Se conservan iguales cortes en las sensibilidades. Cada partición de entrenamiento/evaluación debe contener ambas clases. Estas reglas operacionalizan las marcas bajo revisión; no certifican disponibilidad histórica real.

Se reportan reservas de cada ventana, retenidas, `excluded_labels_not_available`, `excluded_missing_label_recorded_at` y `excluded_recorded_at_or_after_cutoff`, desglosadas también por target; además de intervalos reales, `max_label_recorded_at` y clientes compartidos (solo conteos). La repetición de clientes entre períodos simula clientes nuevos y recurrentes; no demuestra rendimiento exclusivo en clientes nunca vistos.

El experimento base utiliza exactamente las 19 predictoras de `datasets/schema.py`. Un transformer valida el contrato, ignora campos ajenos, canoniza categorías y comparte la ruta de transformación con la futura inferencia. Imputación de mediana y codificación one-hot (incluye clínicas, áreas, booleanos y calendario) se ajustan únicamente con entrenamiento; categorías nuevas se toleran, y columnas completamente nulas conservan su posición. No se utiliza ningún identificador personal, fecha de auditoría, target ni indicador UID como predictor. **`label_recorded_at` se usa solo para particiones y queda fuera de `X`**.

Se entrenan Random Forest, XGBoost y baseline de probabilidad constante igual a la prevalencia de entrenamiento (predicción trivial de clase mayoritaria con el umbral diagnóstico). Los hiperparámetros son fijos antes de ver prueba; pesos de clase se calculan solo con entrenamiento. Semilla predeterminada 42, CPU y hasta dos hilos. No se realiza búsqueda ni ajuste sobre prueba.

Sensibilidades obligatorias: `without_is_fwa`, `without_catalog` (sin áreas/tipos/evaluación médica) y `without_uid_minus_one` (excluye filas candidatas UID -1 de todas las particiones). La exclusión de filas UID -1 **no reconstruye** su contribución en el historial previo de otras citas: el experimento no equivale a retirar esas etiquetas de todo el ETL.

Cada ejecución contiene `experiment.json`, `audit.json`, `review.json`, `validation_comparison.json` y, por modelo/experimento, `model.joblib` y reportes `validation/`. Los metadatos incluyen hash/versiones del dataset, hashes exactos del auxiliar y su manifest, hash del código y modelo, contrato y variables efectivas, hiperparámetros, versiones del entorno, semilla, particiones y métricas. `artifact_format_version` pasa a 2: experimentos anteriores basados en demora global no son compatibles con la evaluación nueva. `experiment.json` se escribe solo al finalizar los doce modelos; una carpeta parcial no es un experimento terminado.

## Evaluar una vez el período reservado

```bash
depi-ml evaluate \
  --csv data/exports/training_dataset_v1.csv \
  --labels data/exports/training_dataset_v1_labels.csv \
  --experiment experiments/phase2-01 \
  --output reports/phase2-test-01 \
  --shap
```

Omite `--shap` si no instalaste `.[explain]`. Solo carga artefactos propios y locales: joblib deserializa objetos Python, no uses modelos de terceros. Comprueba hash del dataset/modelos/código, **mismos bytes del auxiliar y su manifest utilizados para entrenar**, contrato, versiones de dependencias y particiones recalculadas antes de cargar modelos. Mover copias idénticas a otra ruta no cambia su identidad; modificar o reexportar cualquiera de esos archivos sí la cambia y bloquea evaluación. Usa el mismo código y entorno registrado en `experiment.json` para reproducibilidad; SHAP no participa del entrenamiento. Solo se guardan sus importancias agregadas, no valores por cita.

Reportes: ROC-AUC, **PR-AUC como average precision** (no área trapezoidal), precision, recall, F1, matriz `[[TN, FP], [FN, TP]]`, Brier score, curvas ROC/PR con línea de referencia, histograma de probabilidades, diagnóstico de calibración sin recalibrar, permutation importance sobre columnas originales y SHAP sobre columnas codificadas. Las importancias por permutación usan caída de average precision, cinco repeticiones y hasta 2.000 observaciones reproducibles. SHAP usa hasta 200 observaciones de esa muestra; RF explica probabilidades y XGBoost márgenes, por lo que sus magnitudes no se comparan directamente. Variables correlacionadas y predictores redundantes dificultan interpretación; las importancias no son causales.

Las métricas por clínica/período se publican con al menos 30 citas por defecto. En subgrupos de una sola clase se reporta `null` para ROC-AUC y PR-AUC; precision/recall/F1 siguen una convención explícita `zero_division=0`. Si la muestra de permutation importance tiene una sola clase, se registra omisión. Un baseline no necesita importancias.

El umbral 0,5 es diagnóstico y está fijado previamente, no constituye un criterio operativo validado; puede elegirse otro con `--threshold` antes de tocar prueba. No se crean bandas LOW/MEDIUM/HIGH. Calibración, criterios de capacidad/costos y validación prospectiva quedan pendientes. Después de observar prueba, cualquier iteración necesita un nuevo período reservado; comparar sensibilidades en prueba no autoriza seleccionarlas retrospectivamente y afirmar validez independiente.

## Qué compartir para revisión

Primero comparte `methodology.md`, `audit.json`, `monthly.csv`, `clinic_month.csv`, `uid_groups.csv`, `uid_feature_comparison.csv`, `missing_values.csv`, `atypical_cohort.json` y `monthly_rate.png`. Acompaña el SQL completo de construcción y evidencia de temporalidad/etiquetas, revisados para no incluir filas ni credenciales. La plantilla vacía no es evidencia.

Para este ajuste, comparte también `label_availability_report.json` y el `*_labels_quality_report.json` agregado. No compartas el auxiliar CSV: contiene identificadores de citas para correspondencia. La revisión de marcas inconsistentes puede requerir consultas posteriores de solo lectura, pero este ajuste no reinterpreta ni modifica sus valores.

Solo cuando se autorice científicamente el experimento, comparte `experiment.json`, `validation_comparison.json`, `test_comparison.json`, métricas por subgrupo y gráficos/importancias agregadas. No compartas CSV fuente, modelos, `.env` ni registros individuales. Mantén los destinos bajo `reports/` y `experiments/`, ignorados por Git; una ruta personalizada requiere su propia regla de exclusión.

Los artefactos son **experimentales y no listos para producción**. El futuro FastAPI deberá construir features point-in-time de citas futuras desde PostgreSQL y aplicar este contrato/preprocesamiento sin consultar `target`, la tabla de entrenamiento ni información posterior a la predicción.

## Referencias técnicas

El ajuste de transformaciones exclusivamente en entrenamiento sigue las recomendaciones de [scikit-learn sobre leakage y pipelines](https://scikit-learn.org/stable/common_pitfalls.html). La definición reportada de PR-AUC usa [average precision de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). La dependencia OpenMP en macOS se describe en la [instalación oficial de XGBoost](https://xgboost.readthedocs.io/en/stable/install.html). Los agregados SHAP utilizan [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
