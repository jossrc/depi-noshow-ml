# Informe técnico ML — 11 de octubre de 2026

**PYTHON_ML_READY_LOCAL=true**. RF y XGBoost originales cargaron y respondieron
mediante FastAPI real, con comparación, lotes y SHAP verificados.
Estado científico: **EXPLORATORY_NOT_VALIDATED**. Las auditorías de evidencia
histórica y la aprobación del semáforo permanecen parciales.

## Implementación y entorno

Raíz única de código: `depi-noshow-ml`. Rama `main`, commit inspeccionado
`f2f30b5f044cbc86e166da9129efdc721594f3a4`; árbol inicial limpio.
Sin commit/push, acceso a código Node/Next ni sincronización FLOWww/MySQL.

Python 3.12.15; scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.2.6, pandas 2.3.3,
joblib 1.6.0, SHAP 0.52.0, FastAPI 0.143.0, Pydantic 2.14.0, Uvicorn 0.54.0,
httpx 0.28.1, pytest 9.1.1, psycopg 3.3.6. `pip check` pasó. No se instaló ni
actualizó ninguna dependencia. XGBoost nativo cargó correctamente.

Todos los archivos de código/documentación editados están en este repositorio:

| Archivo | Cambio |
| --- | --- |
| src/depi_ml/evaluation/reports.py | Corrige SHAP offline de XGBoost sobre CSR |
| src/depi_ml/infrastructure/ml/shap_inputs.py | Nuevo helper de densificación con semántica de missing preservada |
| src/depi_ml/infrastructure/ml/shap_explainer.py | API reutiliza helper, sin cambiar el score ni su semántica |
| src/depi_ml/db/connection.py | Sesión READ ONLY por defecto, timeout y verificación de identidad/aislamiento |
| tests/test_connection.py | Regresiones de seguridad y ciclo de vida |
| tests/test_offline_shap.py | Regresión numérica SHAP con estimador sintético pequeño |
| tests/api/test_integration_local.py | Usa pin confiable explícito; deja de autopinear el archivo a probar |
| scripts/smoke_ml_api.py | Amplía comprobaciones HTTP; clave y cuerpo por stdin de curl |
| scripts/audit_postgres_readonly.py | Nueva auditoría SQL fija, agregada, acotada y sin PII |
| scripts/verify_ml_validation.py | Reproduce validación original, sin fit/test ni scores individuales |
| docs/future_data_refresh_runbook.md | Procedimiento futuro, sin ejecutar sincronización |
| docs/python_to_backend_handoff.md | Arranque y requisitos concretos para Node |
| docs/ml_python_recovery_20261011.md | Este informe |

Incidencia de runtime: un diagnóstico SHAP temprano sin `MPLCONFIGDIR` provocó
una caché temporal Matplotlib fuera del repositorio, en el directorio temporal
del sistema. Se documenta esa escritura automática; no se afirma que no ocurrió.
Después se fijaron `TMPDIR`, `MPLCONFIGDIR` y `--basetemp` dentro de `reports/`.
No se modificó código externo ni se eliminaron archivos ajenos.

## Artefactos, contrato y modelos

Integridad final: **268 archivos protegidos idénticos**, incluidos los **28
joblib**, CSV, auxiliares, manifests, SQL, `.env`, Compose y contratos públicos.
Esquema predictor, preprocesamiento original y clasificador de riesgo intactos.
OpenAPI y tipos derivados regenerados solo bajo reports coinciden byte a byte.

Pin confiable del manifiesto original, tomado del runbook versionado y verificado:
`ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac`.
Ambos artefactos se verificaron antes de deserializar, con versiones exactas.

| Modelo | SHA-256 completo | Probabilidad de fixture pública sintética |
| --- | --- | ---: |
| random_forest_full_v1 | 4e3c86a170056146cf82a2cc109ae9051b81845608dbd58aae11e43d97492e08 | 0.697640043334 |
| xgboost_full_v1 | 376d3db9d4d32b69784aeb49908ed4f564f95d4528c1f9ff4e766d9fdea505db | 0.915928661823 |

Pipeline completo `contract → preprocess → model`, clase positiva 1 en `[0,1]`.
Los números anteriores **no son resultados de una cita real**. Diferencia
descriptiva 0.218288618489, sin ensemble, ganador ni interpretación de confianza.

Se mantienen exactamente las **19 predictoras** de `PREDICTOR_COLUMNS`, en el
mismo orden y con tipos/nullabilidad/semántica HTTP existentes. Las 213.873 filas
del CSV son compatibles con `PredictionFeatures` al serializar los tipos
PostgreSQL originales: cero errores. La matriz por variable está en el reporte
enlazado abajo; enumera tipos, nulos, evidencia y límites temporales.
Sin predictor de cancelación, cambio de target o modo de scoring actual.
`BOOKING_BASELINE_V1`, `NO_SHOW_CANDIDATE` y snapshot `booking_creation` intactos.

## FastAPI y SHAP comprobados

**23 comprobaciones Uvicorn/curl reales** en `127.0.0.1:8001`:

- 200: live, ready, catálogo, RF individual, XGB individual, compare, explain RF,
  explain XGB, batch single y batch compare.
- 401: credencial ausente en catálogo/explain/batch, incorrecta y duplicada.
- 400 JSON inválido; 409 versión incompatible; 413 payload grande.
- 422 tipo inválido, predictor extra, snapshot current, top_k inválido y lote vacío.

429, 503 y errores saneados se verificaron en escenarios controlados de la suite;
no se provocó saturación o caída del servicio vivo. Se preservan autenticación,
límites, readiness de ambos modelos y activación exclusiva de development.

SHAP RF explica probabilidad: error de aditividad `1.1038259e-10`, tolerancia
`1e-7`. XGB explica log-odds: error `1.1617096e-6`, tolerancia `1e-5`.
Se verificaron base+suma, enlace sigmoid, top_k/orden, factores dependientes de
la entrada y paridad/no mutación de scores. Durante requests, los tests reales
bloquean SQL, `read_csv`, `joblib.load` y `fit`: las peticiones no dependen de DB.

La ruta offline de XGB densificaba ausencias CSR como cero y podía explicar otra
salida. Se corrigió y se compartió la conversión ya correcta de la API. Una
regresión desigual de 60 filas artificiales reprodujo un error de importancia
absoluta 0.546624 antes del arreglo; pasó después. La distinción sparse/dense
está documentada por [XGBoost](https://xgboost.readthedocs.io/en/stable/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data).

`risk_level=null`, `risk_policy_status=pending_validation` y
`operational_use_allowed=false` se conservaron. El smoke terminó únicamente su
propio proceso; **no quedó un servidor persistente iniciado por Codex**.
Comando completo para dejarlo activo: [handoff](python_to_backend_handoff.md).

## Auditoría de datos y PostgreSQL

CSV/manifest/auxiliar originales presentes e íntegros: **213.873 citas, 56.488
clientes, 15 sedes**, 44.320 candidatos positivos, prevalencia 20.722578%.
Cero duplicados, targets/fechas inválidos, anticipaciones negativas o incoherencias
observables de historial. El auxiliar tiene cero fechas ausentes/errores temporales
críticos y **3.410 advertencias** de cierres entre reserva y cita; retraso máximo
de cierre de asistencia 220.854444 días. No se corrigieron ni borraron filas.

Base existente `depi-noshow-db`, rol `admin`. La auditoría impuso sesión
`default_transaction_read_only=on` y verificó identidad, READ ONLY y REPEATABLE
READ antes de consultar negocio, en cada transacción. Timeout local 15 segundos;
SQL fijo/parametrizado y salidas agregadas. Categorías: metadatos, contrato y
calendario, historial as-of/proxies, cabeceras/enlaces/detalles, estados/eventos,
fases de servicios y cancelaciones. No se exportaron filas personales.

Hallazgos comprobados:

- Cero discrepancias de contadores contra el último acumulado con
  `known_at < prediction_at`; cero evento propio/recencia futura. Marcas de
  302.325 asistencias y 57.338 no-shows coinciden con los proxies registrados
  de etiqueta. Esto acredita coherencia de tablas, no conocimiento histórico real.
- Creación coincide en las 213.873 filas; **190.419 cabeceras** tienen modificación
  posterior y 23.454 stamp ausente. No se conoce qué campo cambió en cada edición.
- Las 213.873 filas enlazan actualmente con `training_eligible=false`; revisar
  reglas/versiones antes de usar ese flag para reconstruir población histórica.
- Calendario 1–6 coincide con DOW e ISODOW; no hay domingos que permitan
  discriminar. HTTP exige ISO 1–7 y auditoría offline histórica usa DOW 0–6.
  SQL ETL completo/domingo siguen pendientes; no se cambió semántica sin evidencia.
- **259.045 enlaces**, 259.021 destinos estructuralmente completos, **24 rotos**,
  cero ciclos; **114 detalles huérfanos**. Profundidad máxima observada 1 no prueba
  una cadena completa e inmutable de revisiones intermedias.
- Los pares existentes conservan detalles; 21.749 enlaces tienen diferente número
  de líneas, incluidos destinos ausentes. Muestra determinista de 10.000 enlaces:
  9.991 pares con detalles, 9.008 multisets iguales y **983 distintos**. No es
  muestra representativa ni demuestra una causa específica de edición de servicios.
- Una consulta de firmas sobre toda la tabla agotó 15 s (`57014`); se conservó
  ese fallo y la versión acotada pasó. No se elevó timeout para forzarla.
- **514.075 CANCELLED**, 502.615 resolved_at anteriores a cita. No hay eventos
  CANCELLED inmutables acreditados; no se estimó asociación predictiva de
  cancelaciones previas usando marcas actuales como si fueran conocimiento pasado.

**postgres_write_operations=0; postgres_related_docker_mutations=0;
unrelated_docker_actions=[]**. No se usó Docker, DDL, ETL, migración, ingesta,
exportación nueva o sincronización. La nueva guardia de la conexión oficial se
probó unitariamente; las consultas reales usaron la guardia equivalente del
script de auditoría, con solo SET LOCAL timeout.
Reparar fuente/enlaces/elegibilidad y reconstruir analytics quedan
**BLOCKED_BY_READ_ONLY_POLICY** en esta tarea.

## Métricas efectivamente reproducidas

**TRAINING_REUSED_EXISTING**: sin fit de modelos históricos ni nueva optimización.
Los unit tests sí ajustaron sus pequeños estimadores sintéticos.
Cortes originales por `prediction_at`, seed 42, umbral diagnóstico 0.5.
Train: 130.042 después de purgar 5.725 etiquetas. Validación abril–junio 2026:
**41.427 filas, 9.924 positivos, prevalencia 23.955391%**, después de purgar 4.212
etiquetas (3.629 asistencias, 583 no-shows). 16.965 clientes compartidos
train/validation; no se afirma aislamiento de clientes nunca vistos.

| Modelo | ROC-AUC | AP | Precision | Recall | F1 | Brier | Matriz [[TN,FP],[FN,TP]] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline prior | 0.500000 | 0.239554 | 0 | 0 | 0 | 0.182193 | [[31503,0],[9924,0]] |
| RF original | 0.750229 | 0.512569 | 0.482153 | 0.509069 | 0.495246 | 0.175586 | [[26077,5426],[4872,5052]] |
| XGB original | 0.724693 | 0.473073 | 0.484682 | 0.417674 | 0.448690 | 0.171722 | [[27096,4407],[5779,4145]] |

Métricas originales reproducidas a `1e-12`, conteos/matrices exactos.
Baseline prior reproducido algebraicamente sin fit/deserializar ese baseline.
AP es [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
Log-loss calculado ahora: RF 0.527997994, XGB 0.510490000. Deltas frente a baseline,
subgrupos por sede con N>=30 y resultados mensuales completos están en el JSON.
Subgrupos sin ambas clases conservan métricas AUC/AP null.

Se reutilizaron **34 entradas históricas** de métricas con procedencia y hashes
de modelos verificados: 12 originales y 22 entradas de optimización que incluyen
referencias. Candidatos/sensibilidades no se cargaron ni promovieron. Solo full
con 19 features se reevaluó ahora. No se movieron cortes ni ajustaron umbrales.

**Test reservado sin materialización para modelado, scores o evaluación**.
Las auditorías de integridad recorrieron el original completo según autorización.
No se ejecutó `evaluate` sobre test real ni se falsificó revisión humana.

## Calibración y límites científicos

**NOT_APPROVED: no hay umbrales defendibles aún.** Frente a prevalencia 0.239554,
media de score RF=0.383290 y XGB=0.339290. ECE descriptivo con 10 bins uniformes:
RF=0.144360, XGB=0.100344. Bins de fiabilidad y diagnósticos por período en JSON;
ECE depende de bins/población y no certifica calibración. No se ajustó calibrador
ni se cambió la salida pública de API.

AP RF abril 0.543312 → junio 0.358350; XGB 0.504728 → 0.315443.
Prevalencia, purga y población cambian. La transición comercial junio–julio no
tiene día exacto ni causalidad exclusiva acreditados; no se calculó rendimiento
predictivo posterior sobre test. Reutilizar validación tras optimización introduce
optimismo. Estas cifras no prueban rendimiento final ni ROC-AUC>=0.80.

Pendientes: snapshots/versiones históricas de servicios y catálogo, conocimiento
real de etiquetas/eventos, elegibilidad/definición de no-show, domingo/FWA,
madurez/sesgo de selección, generalización poscambio y período independiente
de calibración. Para semáforo faltan objetivo, costes/capacidad de intervención,
incertidumbre con dependencia por cliente y estabilidad por sede/período.

## QA y rendimiento

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Completa, opt-in real | **490** | **0** | **0** |
| Completa, predeterminado | 484 | 0 | 6 |
| API, opt-in real | 316 | 0 | 0 |
| API inicial, predeterminado | 310 | 0 | 6 |
| Regresiones después | 8 | 0 | 0 |

Antes: cinco fallos reproducidos de guardias de conexión y uno SHAP; JUnit
antes/después conservados. Los seis skips son opt-in apagado, no fallos de carga
en la suite real. Warnings de Starlette/httpx y colores SHAP/Matplotlib
documentados sin ocultarlos. `pip check`, `git diff --check` y `py_compile` pasaron.

Microbenchmark TestClient, cinco muestras artificiales por escenario, un worker:
p95 RF 33.35 ms, XGB 7.98 ms, compare 39.24 ms; SHAP RF 1500.23 ms,
XGB 14.95 ms. Concurrencia moderada 2 y una explicación simultánea. Pico RSS
después de requests 797 MiB; incluye bibliotecas/modelos/explicadores, no memoria
aislada del clasificador. Warm-up y batches 1/8/32 documentados.
No constituye SLA, throughput sostenido ni integración Node/FE.

## Checkpoints, evidencias y reproducción

F0/F1/F2/F5/F6/F7/F9/F10: PASS para el alcance técnico verificado.
F3/F4: **PARTIAL**, estructuras/datos auditados pero sin evidencia histórica
inmutable y revisión humana completa. F8: **PARTIAL**, investigación concluida
sin política aprobada. F11/F12: PASS de documentación, sin ejecutar integración
o sincronización. F13: informe con evidencias y límites; no se declara validación
científica completa. Estados detallados en summary.json.

Carpeta nueva ignorada: `reports/ml-python-recovery-20261011-162033/`:

- [summary.json](../reports/ml-python-recovery-20261011-162033/summary.json): fases, guardias y pendientes.
- [Matriz de 19 variables](../reports/ml-python-recovery-20261011-162033/03_dataset_and_temporal_audit.md).
- [Copias/cancelaciones](../reports/ml-python-recovery-20261011-162033/04_cancellations_reschedules_findings.md).
- [Métricas y procedencia](../reports/ml-python-recovery-20261011-162033/metrics_summary.json).
- [QA y rendimiento](../reports/ml-python-recovery-20261011-162033/07_tests_security_and_performance.md).
- [HTTP final](../reports/ml-python-recovery-20261011-162033/http_smoke_final.json).
- [PostgreSQL agregado](../reports/ml-python-recovery-20261011-162033/postgres_audit_bounded.json).
- Inventario, JUnit, hashes antes/después, auditoría inicial y auxiliar, evidencia
  temporal adicional, métricas históricas y benchmark completos en esa carpeta.

Para iniciar Python y definir las ML_* sin importar credenciales PostgreSQL:
[handoff y arranque](python_to_backend_handoff.md). Node deberá construir snapshots
point-in-time, cadenas de copias/detalles, mapeo exacto, persistencia/idempotencia,
RBAC, cliente con errores/backpressure, endpoints de UI y medición completa.
Esas tareas siguen pendientes en su repositorio, que no se abrió.

[Refresco futuro FLOWww](future_data_refresh_runbook.md): autorización posterior
para origen/réplica/analytics, captura de revisiones/known_at, dataset/auxiliar
nuevos, revisión, entrenamiento reproducible y promoción versionada.

Reproducción desde raíz, .venv compatible y **destinos nuevos**:

```bash
mkdir -p reports/ml-api-runtime/tmp reports/ml-api-runtime/mpl
export TMPDIR="$PWD/reports/ml-api-runtime/tmp"
export MPLCONFIGDIR="$PWD/reports/ml-api-runtime/mpl"
ML_RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest -q \
  --basetemp reports/qa-new/tmp --junitxml reports/qa-new/tests.xml
.venv/bin/python scripts/verify_ml_validation.py \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --manifest-sha256 ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac \
  --output reports/validation-new
.venv/bin/python scripts/audit_postgres_readonly.py \
  --output reports/postgres-audit-new.json
.venv/bin/python scripts/smoke_ml_api.py --extensions --port 8001 \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --manifest-sha256 ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac \
  --output reports/http-new/smoke.json
```

Bind loopback y conexión local pueden requerir aprobación del entorno sandbox.
No hace falta administrar Docker. No reutilizar basetemp preexistente que pytest
pudiera limpiar. Auditoría `--only` acepta categorías fijas para evitar repetir
escaneos. No ejecutar comandos de entrenamiento/evaluate para estos originales
como parte de esta reproducción técnica.
