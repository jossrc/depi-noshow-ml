# Fase 2: análisis y experimentos locales

Todo permanece en `depi-noshow-ml`. No se reconstruye la Fase 1 ni se implementa FastAPI. Los tres comandos nuevos leen únicamente el CSV y archivos locales; no importan configuración de conexión ni ejecutan SQL. El CSV fuente y su manifest se verifican por SHA-256, orden de columnas, registros lógicos, tamaño y versiones de features; permanecen intactos.

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

El contrato no incluye disponibilidad real de etiquetas, versiones de reservas modificadas ni vigencia histórica del catálogo. Por ello se requiere un JSON local de revisión vinculado al hash y versiones del dataset, con responsable, fecha con zona, evidencia documental y cota de demora de etiqueta. El programa valida estructura/declaraciones; la persona revisora debe acreditar la evidencia, no basta con editar estados para superar el control. Si no existe una cota de demora defendible, no entrenar: hace falta una etapa posterior autorizada que incorpore disponibilidad real por cita.

1. **history_point_in_time**: revisar el ETL completo que construyó las 19 features y los acumulados de `ml_history_cumulative_v1`. Acreditar `known_at < prediction_at`, conocimiento real de eventos y exclusión de fila propia/futuros; no usar `client_history_current` ni saldos actuales. Comparar conteos y recencias con eventos fuente. El CSV solo permite coherencia interna y comparación parcial con citas observadas, no prueba el join as-of.
2. **booking_snapshot**: comprobar si la reserva, clínica, horario, duración, sexo y servicios SCHEDULED representan lo que se conocía al reservar. Una fase SCHEDULED por sí sola no demuestra que el registro no se haya editado después. Revisar revisiones/reprogramaciones y evitar líneas PERFORMED/BILLED.
3. **label_availability**: justificar una cota máxima `label_delay_hours`, válida para ambas clases en todos los períodos. Revisar `appointment_outcomes.outcome_resolved_at`, eventos/cierres/tracking y `known_at`: un timestamp de reconstrucción actual o una fecha de cita no equivalen a disponibilidad histórica. El pipeline usa `appointment_at + cota`, que sigue siendo una política externa, no una observación por fila.
4. **catalog_snapshot**: acreditar catálogo as-of o invariancia histórica de áreas/tipos/evaluación médica. `product_catalog_resolution.updated_at` y un catálogo actual no bastan para demostrar confiabilidad retrospectiva.
5. **label_definition**: validar candidatos no-show, conflictos de tickets, tracking UID -1 y exclusiones de usuarios/clínicas. Revisar conversiones `timestamp`/`timestamptz` y reglas FLOWww con la zona real. `target=1` conserva la condición de candidato.
6. **selection_bias**: documentar que la evaluación está condicionada a citas elegibles y excluye cancelaciones anticipadas. Eso puede seleccionar usando un evento posterior a la reserva; no generalizar métricas a todas las reservas ni al futuro servicio sin redefinir la población.
7. **uncertain_semantics**: interpretar `is_fwa`, convención weekday y variables operativas. La sensibilidad sin `is_fwa` y sin catálogo no valida automáticamente su temporalidad.
8. **temporal_label_stability**: explicar cambios por mes/clínica, cobertura y reglas, distinguiendo cambios reales de tracking o etiquetas incompletas. No ocultar caídas abruptas antes de entrenar.

`bds/analytics.sql` contiene estructuras y una vista de clasificación; en la revisión inicial no está la consulta completa que pobló las 19 características. No se ejecuta ese archivo. Cualquier futura consulta a PostgreSQL debe realizarse usando una transacción `READ ONLY` como la que ya ofrece el proyecto.

Copia la plantilla a un archivo ignorado y complétala solo tras revisar evidencia real:

```bash
cp reports/phase2-review-01/methodology_review_template.json methodology_review.local.json
```

En `checks`, historia, reserva, disponibilidad, catálogo, definición de etiquetas y estabilidad temporal requieren `status: "verified"`; selección y semántica requieren `verified` o `acknowledged`, con evidencia y limitaciones en ambos casos. `reviewed_at` usa ISO 8601 con zona; `label_delay_hours` es un número finito no negativo. Una cota de 0 solo es válida con evidencia de disponibilidad inmediata de ambas etiquetas. El código conserva la revisión con los artefactos.

## Entrenar después de superar controles

Estos cortes son un **protocolo candidato**, que debe justificarse frente a las alertas y cobertura observadas antes de ejecutarlo; no resuelven el cambio de etiquetas:

```bash
depi-ml train \
  --csv data/exports/training_dataset_v1.csv \
  --review methodology_review.local.json \
  --validation-start 2026-04-01T00:00:00-05:00 \
  --test-start 2026-07-01T00:00:00-05:00 \
  --test-end 2026-10-01T00:00:00-05:00 \
  --seed 42 \
  --output experiments/phase2-01
```

El comando falla sin revisión acreditada. Divide por **momento de reserva**, no por fecha de cita ni al azar: entrenamiento anterior al inicio de validación, validación hasta el inicio de prueba y prueba hasta `test-end` exclusivo. Purga del entrenamiento las etiquetas cuya disponibilidad estimada no precede al inicio de validación; purga de validación las que no preceden a prueba. En prueba exige madurez anterior al momento de exportación. La política considera reservas con anticipación larga y conserva iguales fechas de corte en las sensibilidades. Cada partición debe tener ambas clases.

Se reportan reservas excluidas por madurez, intervalos reales, disponibilidad máxima estimada, clases y clientes compartidos (solo conteos). La repetición de clientes entre períodos simula clientes nuevos y recurrentes; no demuestra rendimiento exclusivo en clientes nunca vistos.

El experimento base utiliza exactamente las 19 predictoras de `datasets/schema.py`. Un transformer valida el contrato, ignora campos ajenos, canoniza categorías y comparte la ruta de transformación con la futura inferencia. Imputación de mediana y codificación one-hot (incluye clínicas, áreas, booleanos y calendario) se ajustan únicamente con entrenamiento; categorías nuevas se toleran, y columnas completamente nulas conservan su posición. No se utiliza ningún identificador personal, fecha de auditoría, target ni indicador UID como predictor.

Se entrenan Random Forest, XGBoost y baseline de probabilidad constante igual a la prevalencia de entrenamiento (predicción trivial de clase mayoritaria con el umbral diagnóstico). Los hiperparámetros son fijos antes de ver prueba; pesos de clase se calculan solo con entrenamiento. Semilla predeterminada 42, CPU y hasta dos hilos. No se realiza búsqueda ni ajuste sobre prueba.

Sensibilidades obligatorias: `without_is_fwa`, `without_catalog` (sin áreas/tipos/evaluación médica) y `without_uid_minus_one` (excluye filas candidatas UID -1 de todas las particiones). La exclusión de filas UID -1 **no reconstruye** su contribución en el historial previo de otras citas: el experimento no equivale a retirar esas etiquetas de todo el ETL.

Cada ejecución contiene `experiment.json`, `audit.json`, `review.json`, `validation_comparison.json` y, por modelo/experimento, `model.joblib` y reportes `validation/`. Los metadatos incluyen hash/versiones del dataset, hash del código y modelo, contrato y variables efectivas, hiperparámetros, versiones del entorno, semilla, particiones y métricas. `experiment.json` se escribe solo al finalizar los doce modelos; una carpeta parcial no es un experimento terminado.

## Evaluar una vez el período reservado

```bash
depi-ml evaluate \
  --csv data/exports/training_dataset_v1.csv \
  --experiment experiments/phase2-01 \
  --output reports/phase2-test-01 \
  --shap
```

Omite `--shap` si no instalaste `.[explain]`. Solo carga artefactos propios y locales: joblib deserializa objetos Python, no uses modelos de terceros. Comprueba hash del dataset/modelos/código, contrato, versiones de dependencias y particiones recalculadas antes de cargar modelos. Usa el mismo código y entorno registrado en `experiment.json` para reproducibilidad; SHAP no participa del entrenamiento. Solo se guardan sus importancias agregadas, no valores por cita.

Reportes: ROC-AUC, **PR-AUC como average precision** (no área trapezoidal), precision, recall, F1, matriz `[[TN, FP], [FN, TP]]`, Brier score, curvas ROC/PR con línea de referencia, histograma de probabilidades, diagnóstico de calibración sin recalibrar, permutation importance sobre columnas originales y SHAP sobre columnas codificadas. Las importancias por permutación usan caída de average precision, cinco repeticiones y hasta 2.000 observaciones reproducibles. SHAP usa hasta 200 observaciones de esa muestra; RF explica probabilidades y XGBoost márgenes, por lo que sus magnitudes no se comparan directamente. Variables correlacionadas y predictores redundantes dificultan interpretación; las importancias no son causales.

Las métricas por clínica/período se publican con al menos 30 citas por defecto. En subgrupos de una sola clase se reporta `null` para ROC-AUC y PR-AUC; precision/recall/F1 siguen una convención explícita `zero_division=0`. Si la muestra de permutation importance tiene una sola clase, se registra omisión. Un baseline no necesita importancias.

El umbral 0,5 es diagnóstico y está fijado previamente, no constituye un criterio operativo validado; puede elegirse otro con `--threshold` antes de tocar prueba. No se crean bandas LOW/MEDIUM/HIGH. Calibración, criterios de capacidad/costos y validación prospectiva quedan pendientes. Después de observar prueba, cualquier iteración necesita un nuevo período reservado; comparar sensibilidades en prueba no autoriza seleccionarlas retrospectivamente y afirmar validez independiente.

## Qué compartir para revisión

Primero comparte `methodology.md`, `audit.json`, `monthly.csv`, `clinic_month.csv`, `uid_groups.csv`, `uid_feature_comparison.csv`, `missing_values.csv`, `atypical_cohort.json` y `monthly_rate.png`. Acompaña el SQL completo de construcción y evidencia de temporalidad/etiquetas, revisados para no incluir filas ni credenciales. La plantilla vacía no es evidencia.

Solo cuando se autorice científicamente el experimento, comparte `experiment.json`, `validation_comparison.json`, `test_comparison.json`, métricas por subgrupo y gráficos/importancias agregadas. No compartas CSV fuente, modelos, `.env` ni registros individuales. Mantén los destinos bajo `reports/` y `experiments/`, ignorados por Git; una ruta personalizada requiere su propia regla de exclusión.

Los artefactos son **experimentales y no listos para producción**. El futuro FastAPI deberá construir features point-in-time de citas futuras desde PostgreSQL y aplicar este contrato/preprocesamiento sin consultar `target`, la tabla de entrenamiento ni información posterior a la predicción.

## Referencias técnicas

El ajuste de transformaciones exclusivamente en entrenamiento sigue las recomendaciones de [scikit-learn sobre leakage y pipelines](https://scikit-learn.org/stable/common_pitfalls.html). La definición reportada de PR-AUC usa [average precision de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). La dependencia OpenMP en macOS se describe en la [instalación oficial de XGBoost](https://xgboost.readthedocs.io/en/stable/install.html). Los agregados SHAP utilizan [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
