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
