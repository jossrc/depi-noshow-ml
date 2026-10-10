# Verificación local — PROMPT7, 8 de octubre de 2026

Etapas A–C implementadas y verificadas; `FASTAPI_LOCAL_INTEGRATION_READY`. RF/XGB conservan `EXPLORATORY_NOT_VALIDATED`, `calibration_status=not_operationally_validated`, `operational_use_allowed=false` y `risk_level=null`. Etapa D diferida. Cambios en el árbol de trabajo para revisión; no se publicaron modelos ni datos.

Se incorporaron paquetes `api`, `application`, `domain` e `infrastructure`, extras `api`, tests y scripts sintéticos. Se actualizaron README, plantilla de entorno, ignores y dos párrafos de documentación de Fase 2 para separar Fastify de serving. La memoria preexistente `docs/memory/Memory01.md` quedó intacta.

| Ejecución realmente realizada | PASS | FAIL / ERROR | SKIP |
| --- | ---: | ---: | ---: |
| Suite anterior, antes de añadir tests API | 168 | 0 | 0 |
| Suite completa, integración real apagada | 341 | 0 | 3 |
| Suite completa, integración real habilitada | **344** | **0** | **0** |

Hay 168 pruebas originales y 176 nuevas en API. Los tres skips predeterminados son las integraciones opt-in; en la ejecución real se ejecutaron todos. Las últimas suites emitieron un aviso de Starlette 1.7.0 sobre la futura migración de su TestClient de httpx a httpx2; el transporte actual funcionó. No se ocultó el aviso. `pip check` no encontró requisitos rotos.

Comandos de aceptación ejecutados desde la raíz:

```bash
.venv/bin/python -m pip install -e '.[dev,ml,api]'
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/api -q
ML_RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest tests/api -q
ML_RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest -q \
  --junitxml=reports/api-prompt7-20261008/pytest_real.xml
.venv/bin/python -m pytest -q \
  --junitxml=reports/api-prompt7-20261008/pytest_default.xml
.venv/bin/python -m pip check
.venv/bin/depi-ml --help
git diff --check
.venv/bin/python scripts/benchmark_ml_api.py \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --manifest-sha256 ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac \
  --output reports/api-prompt7-20261008/benchmark.json
.venv/bin/python scripts/smoke_ml_api.py \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --manifest-sha256 ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac \
  --output reports/api-prompt7-20261008/http_smoke.json
```

La instalación y el bind loopback requirieron las autorizaciones del entorno restringido; terminaron correctamente después de autorizarlos. El smoke ejecutó internamente `python -m uvicorn depi_ml.api.main:create_app --factory --host 127.0.0.1 --port 8001 --no-access-log --log-config examples/uvicorn_logging.json` y curl, con clave efímera transmitida por stdin y sin `POSTGRES_PASSWORD`. Su proceso terminó al finalizar.

Uvicorn/curl comprobó liveness 200, readiness 200, catálogo 200, RF individual 200, XGB individual 200, compare 200 y catálogo sin clave 401. Los errores 400/404/409/413/422/429/500/503 se comprobaron con TestClient y fakes/casos de fallo. No se inventaron respuestas ni modelos ausentes.

La integración real contrastó scores HTTP contra `predict_proba` del mismo pipeline completo, con tolerancia absoluta `1e-12`; incluyó clínica inédita `999999` y nulos permitidos. Se bloquearon `fit`, SQL, `pandas.read_csv` y recargas joblib durante requests. Una referencia copiada con bytes intencionalmente corruptos se rechazó antes de load; no se alteró el original. Los tests unitarios cubren clase positiva en orden invertido, clases inválidas, comparación sin promedio/intercambio, control de concurrencia real, versiones, symlinks/traversal, hashes, lifespan y secretos.

Se simuló ausencia de FastAPI/Pydantic/Uvicorn/httpx mediante un import hook en un proceso independiente: pytest conservó la colección de las **168 pruebas anteriores** y omitió la carpeta API opcional. Esto es una comprobación de colección, no una segunda ejecución completa de un entorno mínimo.

Identidad verificada de los pipelines originales cargados:

| Model key | SHA-256 completo |
| --- | --- |
| `random_forest_full_v1` | `4e3c86a170056146cf82a2cc109ae9051b81845608dbd58aae11e43d97492e08` |
| `xgboost_full_v1` | `376d3db9d4d32b69784aeb49908ed4f564f95d4528c1f9ff4e766d9fdea505db` |

Manifiesto `experiment.json`: `ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac`. Ambos pertenecen a `full`, comparten el contrato canónico de 19 variables y se cargaron satisfactoriamente con el entorno registrado. No faltaron artefactos para integración local.

Integridad antes/después:

- **463 de 463 archivos anteriores intactos**, incluidos CSV/auxiliar/manifests/revisión/reportes, **28 joblib** (12 originales +16 candidatos), los **27 módulos Python anteriores** y los 12 archivos de tests previos.
- Cero módulos anteriores de ML editados o borrados. `preprocessing.py`: `2b9421ba43a6970932af6991d9149b86877329da0e722f666135f8974c26b03b`.
- Hash agregado del código anterior, reconstruido con los mismos paths/bytes verificados: `6b15844bf2813b9e77ea7f5bed5bf4f8e1b9c007fc29dc84793d854e77c7be0f`, igual antes/después.
- Hash agregado actual de `src/depi_ml`, siguiendo `training.experiments.source_hash`: `fc277b520e53d4bc22dffd26a8ebab3bdab278700fd2a2a08b595f062c9ea27b`. Cambia al añadir módulos de serving; no se reemplazó el hash histórico de entrenamiento del manifiesto.
- Base de datos no consultada ni modificada. Docker/ETL, `.env`, cortes temporales, etiquetas, 19 predictores y ronda de optimización conservados. No se entrenó con datos reales ni se evaluó prueba. La regresión anterior sí usa sus pequeñas fixtures sintéticas de entrenamiento, como antes.

El detalle de hashes y JUnit permanece local e ignorado en `reports/api-prompt7-20261008/`: `protected_before.json`, `verification.json`, `pytest_real.xml`, `pytest_default.xml`, `benchmark.json`, `http_smoke.json`.

Microbenchmark en macOS arm64, Python 3.12.15, un worker y una entrada artificial repetida, 25 observaciones por escenario:

| Escenario | Media ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| HTTP TestClient RF | 44.36 | 33.32 | 86.04 |
| HTTP TestClient XGB | 10.01 | 9.21 | 16.21 |
| HTTP TestClient compare | 40.43 | 38.23 | 48.17 |
| HTTP TestClient compare, concurrencia 2 | 76.96 | 74.22 | 110.25 |
| Pipeline RF directo | 27.67 | 29.63 | 32.69 |
| Pipeline XGB directo | 4.92 | 4.77 | 5.53 |

Carga/lifespan del benchmark: **2.8766 s**; arranque hasta liveness de Uvicorn/curl: **2.6667 s**. Pico RSS del proceso completo antes de carga **51.91 MiB**, después de cargar ambos **580.25 MiB**, después de requests **580.84 MiB**. Incluye bibliotecas, snapshot/verificación y pipelines; no es memoria aislada del clasificador. Hubo actividad de otro proceso de verificación local durante la medición, sin ensayo de carga controlado. La media RF puede superar la comparativa por variación entre escenarios; no inferir una mejora del motor comparativo. No se midió escalado de múltiples workers ni SLA extremo a extremo web/Fastify.

Limitaciones conservadas: ocho controles pendientes; scores no calibrados; sesgo de selección/madurez; predicción al reservar; candidato no-show reconstruido; posible concept drift por transición comercial aproximada junio–julio de 2026, sin fecha exacta ni causalidad exclusiva demostrada. Las tasas de cita junio–septiembre no se convierten en predictor ni en métricas de prueba. La meta ROC-AUC ≥0.80 **sobre prueba reservada** sigue pendiente.

Próximo paso: implementar RiskFeaturesBuilder y PredictionClient en el repositorio Fastify, comprobar snapshots point-in-time y contrato OpenAPI/errores con fixtures sintéticas, y medir latencia completa. E2E Node, interfaz, SHAP local, batch, calibración, piloto y producción no se han demostrado en esta entrega.


## PROMPT8 — Explain, batch y contrato Fastify (10-10-2026)

Estado técnico: **READY_FOR_FASTIFY_INTEGRATION** para prototipo académico.
Estado científico intacto: EXPLORATORY_NOT_VALIDATED, calibración no aprobada,
operational_use_allowed=false, risk_level=null. No certifica producción.
Base limpia c7b5cae891b8195fde125110d034a42588f83f2e, exactamente el commit del
prompt. Cambios sin commit; este informe se añade, sin modificar registros anteriores.

### Alcance y arquitectura comprobados

Dos rutas nuevas: POST /internal/v1/predictions/explain y /batch.
ExplainPrediction y PredictBatch usan el mismo PredictionEngine, registry y
pipelines FeatureContract → ColumnTransformer → estimator. Sin fit ni otra
normalización ni joblib.load por request. Batch secuencial y ordenado bajo un
cupo global; admite 1–32 por defecto, máximo configurado 64. Compare prohíbe
model_key incluso null, devuelve dos objetos completos RF→XGB. Validación
atómica: un item inválido causa 422 y cero inferencias. Fallo intermedio retorna
solo error, nunca 200 parcial. Sin IDs ni archivos ni persistencia.

El catálogo, las cinco rutas originales y **todos** sus schemas mantienen el
contrato: fixture pública PROMPT7 y tests de regresión incluyendo safe fields
para extras. Factory inyectable, ciclo de vida único; SHAP opcional falla por
algoritmo sin bloquear readiness/predicción. No código Node ni base de negocio.

### Matriz HTTP final

| Método | Ruta | Request / response | Auth |
| --- | --- | --- | --- |
| GET | /health/live | 200 proceso vivo | red privada/loopback |
| GET | /health/ready | 200 ambos modelos / 503 MODEL_NOT_READY | red privada/loopback |
| GET | /internal/v1/models | catálogo previo sin cambios | X-Service-API-Key |
| POST | /internal/v1/predictions | 19 features / PredictionResponse completo | X-Service-API-Key |
| POST | /internal/v1/predictions/compare | 19 features / CompareResponse completo | X-Service-API-Key |
| POST | /internal/v1/predictions/explain | PredictionRequest + top_k 1–10 / PredictionResponse + explanation | X-Service-API-Key |
| POST | /internal/v1/predictions/batch | mode single/compare + items / results indexados completos | X-Service-API-Key |

Todas las rutas internas reutilizan auth antes de body parsing, límite ASGI
262144 bytes por defecto, JSON estricto, ausencia de extras/no finitos,
request ID opaco y errores sin valores/tracebacks/PII. Errores comunes
400/401/404/409/413/422/429/500 INFERENCE_FAILED/503 MODEL_NOT_READY.
Explain añade 500 EXPLANATION_FAILED y 503 EXPLANATION_NOT_READY.
No fallback a cero/LOW/otro algoritmo. Docs solo development + flag explícito.

### Evidencia matemática real

SHAP 0.52.0, TreeExplainer raw/tree_path_dependent sin
background adicional; estructura ajustada del árbol, sin CSV/SQL.
RF (1,144,2), clase positiva consultada en classes_; XGB (1,144), margen binario.
144 columnas agrupadas exhaustivamente a 19 por metadatos públicos one-hot,
sin dividir nombres. Suma conservada, ranking determinista por magnitud y
orden canónico para empates, cero neutral; remainder firmado. top_k solo
reduce presentación, aditividad usa las 19 variables.

| Modelo / espacio | Base | Suma agrupada 19 | Reconstrucción | Salida original | Error absoluto |
| --- | --- | --- | --- | --- | --- |
| RF probability/identity | 0.5000000000000007 | 0.1976400434447955 | 0.6976400434447961 | 0.6976400433344135 | 1.1038259e-10 |
| XGB log_odds/sigmoid | 0.004985474050045013 | 2.383286126538906 | 2.388271600588951 | 2.388272762298584 | 1.1617096e-06 |

XGB sigmoid reconstruida 0.9159285710735943,
predict_proba 0.9159286618232727:
diferencia 9.0749678e-08.
Tolerancias: RF aditividad 1e-7; XGB margen 1e-5; enlace margen original 1e-7;
reconstrucción enlazada en integración 1e-6. Scores single/explain/batch ≤1e-10
absolutos, sin cambiar el umbral por conveniencia. RF parallel predict_proba
puede variar ~2e-16 entre llamadas, por orden de suma.

Diagnóstico resuelto: densificar CSR XGB con ceros daba otra salida (suma raw
~1.216 en vez de ~2.388). Se preservan ausencias CSR como NaN y ceros almacenados
como ceros, igual que su inferencia original. No se cambiaron árboles ni parámetros.
Pruebas reales cubren nulos y clinic_id=999999, ausencia de fit/SQL/CSV/load por
request y cambio genuino de score/factores con otra fixture artificial.
Asociaciones locales del modelo, nunca explicación causal de inasistencias.

### Pruebas ejecutadas y warnings preservados

| Comando exacto | PASS | FAIL | SKIP | Warnings / tiempo |
| --- | --- | --- | --- | --- |
| ML_RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest -q (antes de editar) | 344 | 0 | 0 | 1 / 22.36 s |
| .venv/bin/python -m pytest -q --junitxml=reports/api-prompt8-20261010/offline_verified.xml | 478 | 0 | 6 | 1 / 21.68 s |
| ML_RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest -q --junitxml=reports/api-prompt8-20261010/real_verified.xml | 484 | 0 | 0 | 4 / 45.21 s |

140 casos adicionales respecto a la base; se conserva la suite original.
Los seis skips sin opt-in son exactamente tres casos reales anteriores y tres
nuevos; modelos presentes + opt-in no omite incompatibilidad SHAP.
Warnings: StarletteDeprecationWarning de httpx existente y tres
PendingDeprecationWarning de colores Matplotlib usados al importar SHAP.
No filtros ni supresión ni upgrades. El sandbox no permite ~/.matplotlib:
Matplotlib informa uso de caché temporal y construcción de fuentes; se conserva
ese diagnóstico, que contribuye al cold startup medido.

`.venv/bin/python -m pip check`: No broken requirements found.
`git diff --check`: sin errores. CLI --help ejecutado, import API/CLI con SHAP
bloqueado comprobado en subprocess. OpenAPI determinista y tipos derivados
contrastados; ejemplos Swagger/JSON validados por los schemas reales.

### Integridad y límites

455 archivos idénticos byte a byte, y el prefijo histórico de api_verification.md
idéntico: 456 archivos protegidos en total. Incluye 28 joblib, 251 archivos de
experiments, 7 de data/exports, 171 reportes anteriores, 2 SQL, .env, Compose,
Memory01.md y 21 módulos de analysis/datasets/evaluation/training. Inventario
exhaustivo privado: reports/api-prompt8-20261010/protected_before.json;
resultado: protected_after.json. Ningún original borrado/regenerado.

| Artefacto original | SHA-256 verificado |
| --- | --- |
| Manifiesto | ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac |
| RF full | 4e3c86a170056146cf82a2cc109ae9051b81845608dbd58aae11e43d97492e08 |
| XGB full | 376d3db9d4d32b69784aeb49908ed4f564f95d4528c1f9ff4e766d9fdea505db |

SHAP no cambia pyproject: extra existente suficiente, dependencias sin cambios.
.env.example solo añade enable_local_explanations=false y max_batch_items=32;
.env real intacto. .gitignore existente cubre todos los reportes nuevos.
Sin PostgreSQL, CSV real para inferencia, test reservado, Docker ni DOCX/PI1.

Dos cupos globales por worker, una explicación simultánea (segunda → 429),
locks por modelo y una fila densa de ≤1 MiB. Los explicadores se inicializan
una vez desde los predictores cargados. Explain RF puede demorar otro RF
por lock; XGB conserva acceso al segundo cupo. Timeout de red no cancela
cómputo Python iniciado. Límite por worker, no distribuido.

### Smoke y performance reproducibles

Smoke ejecutado con autorización de bind fuera del sandbox:

```bash
.venv/bin/python scripts/smoke_ml_api.py --extensions \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --manifest-sha256 ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac \
  --port 8002 --output reports/api-prompt8-20261010/smoke.json
```

13 comprobaciones: 200 live/ready/models/RF/XGB/compare/explainRF/explainXGB/
batchsingle/batchcompare y 401 models/explain/batch sin key. Clave efímera por
stdin curl, no persistida; puerto verificado libre, proceso propio apagado en
finally. Startup Uvicorn 24.82 s.

Benchmark final sin otras suites simultáneas, entrada artificial, un worker,
TestClient en proceso, cinco muestras por caso; 3 warm-ups single y 1 por
extensión. macOS arm64, 10 CPU lógicas, Python 3.12.15.
Dependencias: FastAPI 0.143.0/Pydantic 2.14.0/Starlette 1.7.0/Uvicorn 0.54.0/
httpx 0.28.1; numpy 2.2.6/pandas 2.3.3/sklearn 1.9.1/XGB 3.4.1/joblib 1.6.0/SHAP 0.52.0.

```bash
.venv/bin/python scripts/benchmark_ml_api.py --extensions \
  --reference experiments/phase2-exploratory-prompt5-20261008 \
  --manifest-sha256 ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac \
  --requests 5 --output reports/api-prompt8-20261010/benchmark_verified.json
```

| Escenario | n | p50 ms | p95 ms |
| --- | --- | --- | --- |
| random_forest_full_v1 | 5 | 21.57 | 22.42 |
| xgboost_full_v1 | 5 | 6.23 | 6.63 |
| compare | 5 | 27.09 | 27.78 |
| compare_concurrency_2 | 5 | 62.00 | 86.95 |
| explain_rf | 5 | 1437.95 | 1449.19 |
| explain_xgb | 5 | 12.05 | 12.64 |
| batch_single_1 | 5 | 31.28 | 46.57 |
| batch_single_8 | 5 | 188.77 | 209.73 |
| batch_single_32 | 5 | 717.10 | 774.34 |
| batch_compare_1 | 5 | 27.93 | 37.23 |
| batch_compare_8 | 5 | 212.99 | 224.59 |
| batch_compare_32 | 5 | 903.44 | 924.93 |
| explain_rf_with_xgb_concurrency_2 | 5 | 1438.09 | 1440.99 |
| xgb_with_explain_rf_concurrency_2 | 5 | 13.87 | 14.27 |

Cold startup 19.74 s, inicialización/import SHAP
17.58 s. Pico RSS acumulativo antes
52.38 MiB, modelos solos
583.02 MiB, con explicadores
814.20 MiB, tras requests
816.44 MiB. Incluye importaciones,
conversión de árboles y cachés, no solo buffers de atribuciones. No es RSS estable
ni medición de allocations SHAP aisladas. Otro worker duplica recursos.

Cinco muestras y percentiles interpolados describen esta máquina/fixture,
no SLA ni carga sostenida ni Next.js/Fastify. Solicitar SHAP solo en detalle;
Fastify deberá medir timeouts, circuit breaker y backpressure extremo a extremo.
No timeout que finja cancelar threads ni sobrecarga ilimitada.

### Transferencia y pendientes científicos

[Runbook Swagger/curl](api_operations.md), [contrato HTTP](api_ml.md),
[arquitectura C4](architecture_ml.md), [Fastify](fastify_integration_contract.md),
[OpenAPI](contracts/ml-internal-v1.openapi.json),
[TS](contracts/ml-internal-v1.types.ts). Cuatro ejemplos públicos completos
prediction_explain_rf/xgb y prediction_batch_single/compare.json en examples.
Se exportó contrato con `.venv/bin/python scripts/export_ml_contract.py`.

HU activas 01/02/03/05/06/08/10/11/12/13/15/17/18/21/22;
retiradas 04/07/09/14/16/19/20. booking_lead_days se conserva. Backend Fastify
pendiente: RiskFeaturesBuilder as-of, PredictionClient, RBAC, CRUD/importación,
idempotencia, persistencia auditada y UI. No se implementa aquí.

Ocho controles pendientes: history_point_in_time, booking_snapshot,
label_availability, catalog_snapshot, label_definition, selection_bias,
uncertain_semantics, temporal_label_stability. Calibración, bandas y desempeño
por grupos/sedes también pendientes. Etiqueta candidata reconstruida, sesgo
por elegibilidad/purga; prueba sin materializar/evaluar y AUC académica ≥0.80
en test sin acreditar. Explain/batch no promocionan ninguno de los modelos.

Política de compra previa de bono aproximadamente junio–julio 2026; excepción
presencial el mismo día con disponibilidad. Posible concept drift: tasas por
mes de cita junio 20.92%, julio 7.15%, agosto 4.78%, septiembre 2.58%, compatibles
con la política, sin causalidad exclusiva probada. No confundir con prevalencia
por mes de reserva. No fecha exacta inventada ni predictor nuevo; cortes,
etiquetas y ronda de hiperparámetros intactos. Generalización postcambio pendiente.

Criterios A–E de PROMPT8 comprobados: regresión/integridad, SHAP real para ambos,
batch/paridad/admisión, seguridad y transferencia con smoke/contratos/tests.
Sin bloqueos residuales de implementación Python. Uso operativo y construcción
Fastify siguen pendientes y explícitamente fuera de esta ampliación.
