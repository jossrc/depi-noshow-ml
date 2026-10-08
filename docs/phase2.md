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

Fechas faltantes se conservan como vacías y se reportan por clase. La política temporal `outcome_aware_v2` distingue:

| Resultado / condición | Clasificación y acción |
| --- | --- |
| Asistencia (`target=0`), `label_recorded_at <= prediction_at` | Error crítico; bloquea el uso del auxiliar |
| Asistencia, `prediction_at < label_recorded_at < appointment_at` | Advertencia operativa; conserva la fila y aplica los cortes ordinarios |
| No-show (`target=1`), `label_recorded_at < appointment_at` | Error crítico; bloquea el uso del auxiliar |
| Cualquier clase, `label_recorded_at > source_snapshot_at` | Error crítico; bloquea incluso si también cumple la condición de cierre temprano |

Una marca igual a la cita se permite en ambas clases; una marca igual a la instantánea no es posterior. FLOWww permite cierres manuales: pueden existir pruebas o errores administrativos, pero no se atribuye una causa individual sin evidencia. Las advertencias no certifican disponibilidad real ni verifican controles metodológicos.

El exportador conserva las marcas críticas sin corregirlas en un **auxiliar de auditoría bloqueado**, escribe el reporte y devuelve código 1; no permite usarlas en entrenamiento/evaluación. Identificadores duplicados, correspondencia/resultado incorrectos, fechas no interpretables/no finitas o cambio del CSV en PostgreSQL detienen la publicación. Una revisión humana no anula esos controles críticos.

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

Sin los tres cortes, el comando valida solo archivos; si se indican cortes deben proporcionarse los tres. `--labels-manifest` permite especificar otro manifest; por defecto usa `<nombre_auxiliar>_manifest.json`. Produce `label_availability_report.json` y `methodology_review_template.json` en una carpeta nueva; rechaza una carpeta ya existente. Ambos son agregados, sin citas individuales; **todos los controles de la plantilla siguen en `pending`**. La validación se recalcula desde los CSV bajo la política vigente; la clasificación histórica del manifest y del reporte de calidad auxiliar se conserva sin cambios.

El reporte separa `critical_temporal_errors` y `operational_warnings`, con conteos totales y por clase. El campo compatible `inconsistent_dates` cuenta únicamente filas críticas; `recorded_before_appointment` permanece descriptivo e incluye advertencias. Los conteos por motivo crítico pueden solaparse, pero `critical_temporal_errors.rows` cuenta cada fila una sola vez; las filas críticas no se cuentan como advertencias. Si hay errores críticos, conserva el reporte, devuelve código 1 y no genera una vista de particiones. Con advertencias solamente devuelve 0, emite un aviso y permite previsualizar conteos; `training_status` continúa en `pending_human_review`. La vista no exige ambas clases y reporta particiones insuficientes; `train` sí exige ambas clases.

La primera extracción real verificó las 213.873 citas y detectó 3.410 marcas anteriores a la cita, todas en asistencias, sin fechas ausentes. La política anterior las clasificó como críticas. PROMPT4 aporta la auditoría de cierres frente a la reserva y solicita distinguirlas como advertencias cuando sean estrictamente posteriores a `prediction_at`. La revalidación local del 8 de octubre de 2026 confirma **0 filas temporalmente críticas y 3.410 advertencias operativas**, sin eliminar filas ni reexportar. El nuevo reporte está en `reports/phase2-labels-prompt4-20261008/label_availability_report.json`; los seis artefactos originales (ambos CSV, manifests y reportes de calidad) conservan sus hashes. El cierre tardío máximo registrado fue aproximadamente 221 días. Estos resultados corresponden al hash del CSV auditado; deben revisarse en el reporte local, sin desplazamiento horario ni recorte automático. El contexto de PROMPT3 declara verificado `known_at < prediction_at`; esa declaración y el contexto operativo de PROMPT4 no se convierten automáticamente en evidencia de revisión firmada.

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
3. **label_availability**: validar `label_recorded_at` para ambas clases, la interpretación Lima del cierre manual, ediciones retroactivas y fecha de ingestión. Revisar `outcome_resolved_at`, eventos/cierres/tracking y `known_at`: una marca registrada no acredita cuándo fue visible al personal/sistema. Resolver anomalías críticas y revisar/documentar advertencias operativas, sin atribuir causas individuales no demostradas. Documentar la exclusión de marcas ausentes/tardías por corte, que puede seleccionar diferencialmente asistencias y no-shows.
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

## Entrenamiento exploratorio explícito

PROMPT5 autoriza una primera ejecución sobre entrenamiento y validación aun con revisión metodológica pendiente. Se activa exclusivamente con `--exploratory`; sin esta opción el bloqueo metodológico continúa vigente. No se modifican datos, predictoras, ETL, PostgreSQL ni estados de la plantilla.

```bash
.venv/bin/depi-ml train --exploratory \
  --csv data/exports/training_dataset_v1.csv \
  --labels data/exports/training_dataset_v1_labels.csv \
  --review reports/phase2-labels-prompt4-20261008/methodology_review_template.json \
  --validation-start 2026-04-01T00:00:00-05:00 \
  --test-start 2026-07-01T00:00:00-05:00 \
  --test-end 2026-10-01T00:00:00-05:00 \
  --seed 42 \
  --output experiments/phase2-exploratory-prompt5-20261008
```

La carpeta de salida debe ser nueva. `--review` es opcional en este modo: si falta, se registra una revisión sin responsable/fecha y con los ocho controles pendientes. Si se proporciona, se verifica su vinculación al dataset y auxiliar, se conserva una copia de sus declaraciones y su hash, sin modificar el original ni certificar evidencia. Las declaraciones sin firma/evidencia suficiente se registran como pendientes. Esto no permite entrenamiento definitivo.

Siguen siendo obligatorios hashes, contrato, tipos, correspondencia exacta del auxiliar, resultados, fechas críticas y coherencia observable del historial. Se verifican las fuentes nuevamente al terminar. Las verificaciones técnicas recorren el CSV completo para detectar inconsistencias; los análisis descriptivos de etiquetas y el modelado se limitan a reservas anteriores a `test_start`. Se exige ambas clases en entrenamiento y validación, purgando etiquetas ausentes o no anteriores al corte; los cierres manuales entre reserva y cita siguen siendo advertencias. La prueba solo tiene conteos de reserva/integridad temporal, su dataframe no se materializa y no se usa en ajuste, predicciones, métricas, importancias ni selección. `evaluate` rechaza cualquier artefacto marcado exploratorio antes de cargar modelos o crear particiones de prueba; aportar después una revisión no habilita su evaluación.

Se ajustan baseline, Random Forest y XGBoost con los mismos hiperparámetros y cuatro experimentos existentes (`full`, `without_is_fwa`, `without_catalog`, `without_uid_minus_one`), dando doce modelos. El preprocesamiento y los pesos se estiman solo con entrenamiento. Se utiliza semilla 42 y umbral diagnóstico fijo 0,5; no se ajustan umbrales operativos ni se selecciona modelo definitivo.

`experiment.json`, `audit.json`, `review.json`, las métricas y los reportes registran `EXPLORATORY_NOT_VALIDATED`. Los metadatos incluyen controles pendientes, instrucciones/limitaciones, identidad del código/datos/modelos, hiperparámetros, versiones y cortes. Cada carpeta `validation/` contiene `report_context.json`, métricas, curvas, diagnósticos agregados e importancias; CSV y gráficos también identifican la condición exploratoria. Las comparaciones de los doce modelos están en `validation_comparison.json`, `.csv` y `.md`, siempre con baseline y ROC-AUC, PR-AUC (average precision), recall, precision, F1 y Brier. El archivo `.md` incluye las limitaciones. `experiment.json` se escribe únicamente al finalizar todos los modelos y verificar las fuentes.

Los resultados no acreditan disponibilidad histórica real de features/etiquetas, snapshots de reserva/catálogo, validez de candidatos no-show, semántica operativa ni estabilidad temporal. La población sigue condicionada por elegibilidad/cancelaciones y la purga por madurez puede introducir selección diferencial. Retirar filas UID -1 no reconstruye el historial del ETL. Pesos de clase pueden distorsionar probabilidades; Brier y la curva de confiabilidad son diagnósticos, sin calibración operativa. La comparación corresponde a validación y no prueba rendimiento sobre el período reservado o producción.

## Una única ronda acotada de hiperparámetros (PROMPT6)

```bash
.venv/bin/depi-ml optimize-exploratory \
  --csv data/exports/training_dataset_v1.csv \
  --labels data/exports/training_dataset_v1_labels.csv \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --review reports/phase2-labels-prompt4-20261008/methodology_review_template.json \
  --output experiments/phase2-optimization-prompt6-20261008
```

La salida debe ser nueva y separada de la referencia. Los cortes se recuperan exclusivamente del experimento original; no se ofrecen opciones para cambiar fechas, semilla ni umbral. Se conservan los controles técnicos, incluyendo purga por `label_recorded_at`, correspondencia exacta, integridad y fechas críticas. La prueba no se materializa. Se verifican las versiones, hashes, features y parámetros de los modelos originales locales antes de deserializar; se reproduce su matriz de confusión y las seis métricas de validación antes de ajustar cualquier candidato. El código nuevo de optimización puede diferir del hash global original: se registran ambos hashes y la equivalencia de métricas, sin alterar los artefactos previos. El preprocesamiento no se modifica.

La ronda está fijada en `training/optimization.py`, sin búsqueda adaptativa ni early stopping:

| Algoritmo | Configuración | Cambios frente al original |
| --- | --- | --- |
| Random Forest | `rf_leaf20` | `min_samples_leaf=20` |
| Random Forest | `rf_features03` | `max_features=0.3` |
| Random Forest | `rf_regularized` | Hoja 20, fracción de variables 0.3, profundidad máxima 16 |
| XGBoost | `xgb_shallow` | 300 árboles, profundidad 3, `min_child_weight=5`, lambda 5 |
| XGBoost | `xgb_depth5` | 300 árboles, profundidad 5, `min_child_weight=5`, lambda 5 |
| XGBoost | `xgb_deep_regularized` | 400 árboles, profundidad 6, aprendizaje 0.03, peso mínimo 10, lambda 10, gamma 0.1 |
| XGBoost | `xgb_slow` | 400 árboles, aprendizaje 0.03, peso mínimo 5, lambda 5 |
| XGBoost | `xgb_unweighted` | 300 árboles, peso mínimo 5, lambda 5, `scale_pos_weight=1` |

Se ejecutan ambos algoritmos en `full` y `without_catalog`: 16 modelos nuevos. Contando la configuración original, hay cuatro RF y seis XGBoost por experimento, por debajo del máximo de ocho. Las seis referencias (incluidos dos baselines) se reevaluúan exclusivamente en validación sin volver a ajustarlas. La semilla es 42 y el umbral diagnóstico 0.5; modificar el umbral no cambia ROC-AUC/AP, calculados con probabilidades.

`plan.json` registra la ronda, criterios y contexto antes de los ajustes. `validation_comparison.json`/`.csv` incluye las seis métricas para referencias y candidatos. `validation_monthly.json`/`.csv` y `optimization_report.md` incluyen todas las métricas, N y prevalencia por **mes de reserva (`prediction_at`) en America/Lima**; esto difiere de la agrupación por fecha de cita del reporte previo y evita confundir citas posteriores con reservas de prueba. Hay gráficos mensuales de AP y Brier por algoritmo/experimento. Se conservan parámetros efectivos, hashes de cada modelo y referencias, estados pendientes y límites comerciales. `experiment.json` solo se publica tras completar toda la ronda y comprobar que fuentes, referencia, código y plan mantienen sus hashes; `evaluate` rechaza el resultado exploratorio.

La relevancia se define **antes de observar candidatos**, como criterio descriptivo: ganancia absoluta de AP >=0.01 frente al mismo algoritmo original, deterioro ROC-AUC <=0.002, aumento Brier <=0.002 y caída AP por mes <=0.01. No es un umbral clínico/operativo ni prueba de significancia estadística. Se muestran todos los candidatos; no se promueve un modelo definitivo ni se ejecuta otra ronda si las diferencias son pequeñas. Reutilizar validación para seleccionar hiperparámetros introduce optimismo: las diferencias no tienen confirmación independiente ni intervalos de incertidumbre.

### Contexto comercial aportado por el usuario

Antes de la transición se permitía reservar sin pagar y abonar al llegar. Después se exigió generalmente comprar un bono antes de reservar; permanece la excepción de atención presencial el mismo día con disponibilidad. La implementación se sitúa aproximadamente entre **junio y julio de 2026**, sin día exacto. No se inventa una fecha, no se modifica ningún corte/etiqueta y esta información no se usa como predictor.

La auditoría previa aportada por el usuario indicó tasas de candidatos no-show de junio 20.92%, julio 7.15%, agosto 4.78% y septiembre 2.58%. Se conservan como **antecedentes externos declarados**, sin recalcular la prueba ni generar métricas predictivas de julio-septiembre. Son compatibles con un cambio de distribución/comportamiento (*concept drift*), pero no demuestran causalidad exclusiva ni separan cambios de mezcla de clientes/clínicas, tracking, cobertura, madurez de etiquetas o selección de atención presencial. Los porcentajes pueden corresponder a otra base temporal/población que la validación purgada; no se exige que coincidan.

La validación de abril-junio incluye junio como posible transición y no estima rendimiento en el régimen comercial posterior. Las tasas y métricas mensuales deben interpretarse conjuntamente; una mejor AP global no demuestra estabilidad temporal ni generalización tras la política. Los ocho controles metodológicos y la calibración operativa siguen pendientes.

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

Sin `--exploratory`, el comando falla sin auxiliar técnicamente coherente y revisión acreditada. Divide por **momento de reserva**, no por fecha de cita ni al azar, y mantiene el orden cronológico por `prediction_at`:

| Partición | Reservas | Marca registrada de etiqueta |
| --- | --- | --- |
| Entrenamiento | `prediction_at < validation_start` | `label_recorded_at < validation_start` |
| Validación | `validation_start <= prediction_at < test_start` | `label_recorded_at < test_start` |
| Prueba | `test_start <= prediction_at < test_end` | `label_recorded_at < exported_at` del **dataset original** |

Las marcas iguales al corte se excluyen. No se sustituye la fecha original por la fecha posterior del auxiliar. Las fechas ausentes se excluyen en memoria, sin imputación ni alteración del CSV. Los cierres días/meses después de la atención se evalúan por su marca real registrada y no por una demora global. Se conservan iguales cortes en las sensibilidades. Cada partición de entrenamiento/evaluación debe contener ambas clases. Estas reglas operacionalizan las marcas bajo revisión; no certifican disponibilidad histórica real.

Se reportan reservas de cada ventana, retenidas, `excluded_labels_not_available`, `excluded_missing_label_recorded_at` y `excluded_recorded_at_or_after_cutoff`, desglosadas también por target; además de intervalos reales, `max_label_recorded_at` y clientes compartidos (solo conteos). `operational_warning_rows_in_window` y `retained_operational_warning_rows` muestran las advertencias por partición: no son un motivo de exclusión. La repetición de clientes entre períodos simula clientes nuevos y recurrentes; no demuestra rendimiento exclusivo en clientes nunca vistos.

El experimento base utiliza exactamente las 19 predictoras de `datasets/schema.py`. Un transformer valida el contrato, ignora campos ajenos, canoniza categorías y comparte la ruta de transformación con la futura inferencia. Imputación de mediana y codificación one-hot (incluye clínicas, áreas, booleanos y calendario) se ajustan únicamente con entrenamiento; categorías nuevas se toleran, y columnas completamente nulas conservan su posición. No se utiliza ningún identificador personal, fecha de auditoría, target ni indicador UID como predictor. **`label_recorded_at` se usa solo para particiones y queda fuera de `X`**.

Se entrenan Random Forest, XGBoost y baseline de probabilidad constante igual a la prevalencia de entrenamiento (predicción trivial de clase mayoritaria con el umbral diagnóstico). Los hiperparámetros son fijos antes de ver prueba; pesos de clase se calculan solo con entrenamiento. Semilla predeterminada 42, CPU y hasta dos hilos. No se realiza búsqueda ni ajuste sobre prueba.

Sensibilidades obligatorias: `without_is_fwa`, `without_catalog` (sin áreas/tipos/evaluación médica) y `without_uid_minus_one` (excluye filas candidatas UID -1 de todas las particiones). La exclusión de filas UID -1 **no reconstruye** su contribución en el historial previo de otras citas: el experimento no equivale a retirar esas etiquetas de todo el ETL.

Cada ejecución contiene `experiment.json`, `audit.json`, `review.json`, `validation_comparison.json`/`.csv`/`.md` y, por modelo/experimento, `model.joblib` y reportes `validation/`. Los metadatos incluyen hash/versiones del dataset, hashes exactos del auxiliar y su manifest, hash del código y modelo, contrato y variables efectivas, hiperparámetros, versiones del entorno, semilla, particiones y métricas. `artifact_format_version` pasa a 2: experimentos anteriores basados en demora global no son compatibles con la evaluación nueva. `experiment.json` se escribe solo al finalizar los doce modelos; una carpeta parcial no es un experimento terminado.

## Evaluar una vez el período reservado

Este comando exige un entrenamiento con revisión acreditada y **rechaza experimentos exploratorios**. PROMPT5 mantiene la prueba reservada: no ejecutar esta etapa para los resultados exploratorios.

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
