# Operación local de la demo FastAPI

Ejecutar desde la raíz, Python 3.12 y entorno existente. No copiar ni sobrescribir `.env`, consultar PostgreSQL, regenerar CSV, entrenar, optimizar, evaluar prueba ni recrear Docker para este servicio.

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,ml,api]'
python -m pytest -q
```

Versiones verificadas localmente: Python 3.12.15; pandas 2.3.3; NumPy 2.2.6; scikit-learn 1.9.1; XGBoost 3.4.1; joblib 1.6.0; FastAPI 0.143.0; Pydantic 2.14.0; Starlette 1.7.0; Uvicorn 0.54.0; httpx 0.28.1. El extras `api` mantiene rangos compatibles; no hay política de lock preexistente. Operaciones debe congelar la resolución probada para cada release. **No actualizar bibliotecas ML para forzar carga de joblib:** el verificador exige versiones exactas del experimento y Python major/minor iguales. En macOS XGBoost necesita la instalación OpenMP ya usada por Fase 2; no hace falta reinstalarla si carga correctamente.

Configurar en el entorno de la terminal, sin secretos escritos en el historial. La API **no carga `.env`** ni necesita `POSTGRES_PASSWORD`. Para una clave efímera local:

```bash
export ML_API_ENV=development
export ML_API_MODE=exploratory_demo
export ML_ALLOW_EXPERIMENTAL_INFERENCE=true
export ML_MODEL_REFERENCE_DIR=experiments/phase2-exploratory-prompt5-20261008
export ML_MODEL_MANIFEST_SHA256=ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac
export ML_DEFAULT_MODEL_KEY=random_forest_full_v1
export ML_MAX_REQUEST_BYTES=262144
export ML_MAX_CONCURRENT_PREDICTIONS=2
export ML_EXPOSE_DOCS=true
export ML_API_SERVICE_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

El pin mostrado corresponde al `experiment.json` original local **verificado el 8-10-2026**, no a un manifiesto cualquiera. En otra máquina recuperar una copia privada confiable e idéntica; no limitarse a calcular un nuevo hash de un archivo desconocido. Los 12 modelos originales y 16 candidatos no están en Git. Nunca usar uploads, modelos externos ni un clasificador extraído sin preprocesamiento. SHA detecta cambios respecto de una referencia confiable, pero no hace segura una deserialización maliciosa.

Arrancar ambos modelos, sin importar credenciales de BD:

```bash
uvicorn 'depi_ml.api.main:create_app' --factory --host 127.0.0.1 --port 8001 \
  --no-access-log --log-config examples/uvicorn_logging.json
```

La configuración de logging genera eventos con campos, request ID, ruta de contrato, duración y modelo/hash. Deshabilita el access log genérico, que podría conservar URLs arbitrarias. No habilitar logging de cuerpos/headers, debug HTTP, `curl -v` ni trazas con claves. Sin este log-config, los loggers de biblioteca siguen la configuración del proceso anfitrión.

Para probar **en otra terminal**, definir la misma clave mediante entrada oculta, sin escribirla en el comando: `read -r -s ML_API_SERVICE_KEY` y luego `export ML_API_SERVICE_KEY`. También se puede ejecutar el curl en la terminal configurada usando Uvicorn en otra sesión. No imprimir la variable ni pasarla como argumento literal. El siguiente patrón envía la cabecera por stdin de configuración de curl:

```bash
curl --silent --show-error http://127.0.0.1:8001/health/live
curl --silent --show-error http://127.0.0.1:8001/health/ready

python -c 'import json,os; print("header = " + json.dumps("X-Service-API-Key: " + os.environ["ML_API_SERVICE_KEY"]))' | \
  curl --silent --show-error --fail-with-body --max-time 10 --config - \
  http://127.0.0.1:8001/internal/v1/models

python -c 'import json,os; print("header = " + json.dumps("X-Service-API-Key: " + os.environ["ML_API_SERVICE_KEY"]))' | \
  curl --silent --show-error --fail-with-body --max-time 10 --config - \
  --header 'Content-Type: application/json' --header 'X-Request-ID: synthetic-demo-001' \
  --data-binary @examples/prediction_single.json \
  http://127.0.0.1:8001/internal/v1/predictions

python -c 'import json,os; print("header = " + json.dumps("X-Service-API-Key: " + os.environ["ML_API_SERVICE_KEY"]))' | \
  curl --silent --show-error --fail-with-body --max-time 10 --config - \
  --header 'Content-Type: application/json' --header 'X-Request-ID: synthetic-demo-002' \
  --data-binary @examples/prediction_compare.json \
  http://127.0.0.1:8001/internal/v1/predictions/compare
```

Los dos JSON de `examples/` son públicos y artificiales, sin secretos ni personas reales. Para XGB cambiar únicamente `model_key` a `xgboost_full_v1` en una copia sintética bajo `fixtures_locales/` (ignorado). No editar el contrato ni completar inputs desde el CSV histórico.

Si no hay clave/activación/referencia, liveness sigue vivo y readiness devuelve 503; rutas internas sin clave devuelven 401. Clave configurada requiere 32–512 caracteres ASCII imprimibles, sin espacios extremos. Configuración inválida detiene la factory con mensaje saneado. Un artefacto ausente/alterado, versión incompatible o biblioteca nativa no cargable deja readiness e inferencias en 503 y registra un código estático en startup. No hay fallback RF ante fallo XGB. No publicar estos artefactos en staging/production: la demo solo se habilita en development. `ML_REQUIRE_BOTH_MODELS` no es una opción: los dos son obligatorios.

Pruebas y mediciones opt-in, sin CSV/SQL:

```bash
ML_RUN_REAL_INTEGRATION=1 python -m pytest -q
ML_RUN_REAL_INTEGRATION=1 python -m pytest tests/api -q
python scripts/benchmark_ml_api.py --reference "$ML_MODEL_REFERENCE_DIR" \
  --manifest-sha256 "$ML_MODEL_MANIFEST_SHA256" --output reports/api-demo-new/benchmark.json
python scripts/smoke_ml_api.py --reference "$ML_MODEL_REFERENCE_DIR" \
  --manifest-sha256 "$ML_MODEL_MANIFEST_SHA256" --output reports/api-demo-new/http_smoke.json
```

El benchmark usa TestClient, 25 solicitudes por escenario y dos threads para concurrencia moderada. Mide carga, p50/p95 y pico RSS del proceso completo; no sustituye integración Fastify ni SLA. El smoke crea una clave aleatoria no persistida, arranca Uvicorn en loopback, llama curl, prueba RF/XGB/compare y 401, y termina **su propio** proceso en `finally`. Puerto 8001 debe estar libre. En entornos restringidos puede requerir autorización para bind local. Los destinos deben ser nuevos; ambos scripts rechazan sobrescrituras.

Rotación: generar nueva clave en gestor de secretos/entorno, actualizar futuro Fastify y reiniciar servicio coordinadamente; no hay endpoint de configuración ni período de doble clave implementado. Un nuevo modelo requiere revisión offline, manifiesto/artefactos versionados, pruebas y release explícito; el catálogo actual fija los originales. No sobrescribir joblib cargados. Rollback a referencia y resolución de dependencias anterior, seguido de readiness y paridad. Mantener modelos de solo lectura con control de acceso; usar usuario no root cuando se añada contenedor API. Docker/ETL actuales no se modifican en esta entrega.

Cada worker carga ambos pipelines y duplica su RAM. La configuración local usa un worker; múltiples workers no se benchmarkearon. El máximo de concurrencia es por worker, no un rate limit distribuido. El consumidor puede limitar reintentos y usar circuit breaker; el timeout de red no interrumpe una inferencia en thread ya admitida. Añadir aislamiento de procesos/recursos antes de un despliegue con cargas no controladas. TLS/red privada pertenecen a la infraestructura futura, no a certificados inventados en esta demo.

## Operación de explain y batch (PROMPT8)

El extra existente `explain = ["shap>=0.46,<1"]` es suficiente; SHAP **0.52.0**
verificado con ambos originales sin cambiar dependencias. Contrastar entorno
con manifiesto antes de instalar extras; no actualizar ML para silenciar warnings.

Añadir al entorno configurado arriba y arrancar un worker:

```bash
export ML_ENABLE_LOCAL_EXPLANATIONS=true
export ML_MAX_BATCH_ITEMS=32
uvicorn 'depi_ml.api.main:create_app' --factory --host 127.0.0.1 --port 8001 \
  --no-access-log --log-config examples/uvicorn_logging.json
```

Flags nuevos estrictos: true/false literal en entorno (default false);
batch entero decimal 1–64 (default 32). ApiSettings rechaza strings y bool
usados como enteros. Barrera development, activación experimental, pin y
clave siguen obligatorios. Sin PostgreSQL ni lectura de .env.

Inicializa explicadores una vez desde los predictores cargados. Fallo opcional
por modelo: 503 `EXPLANATION_NOT_READY`; readiness/single/compare/batch disponibles.
503 `MODEL_NOT_READY` es fallo de modelos/barrera demo. No sustituye XGB por RF.
Excepción de explicación por request: 500 `EXPLANATION_FAILED` saneado.

Swagger: abrir `http://127.0.0.1:8001/docs` con env development y expose_docs true.
En **Authorize**, introducir manualmente la clave desde una entrada/gestor seguro;
no incorporarla a OpenAPI, JSON, capturas o historial. Seleccionar ejemplo
RF/XGB en explain (top_k 1–10), single/compare en batch. Comprobar hash corto,
estados, factores, índice y request ID. La clave efímera del smoke no se reutiliza.

Curl completo, variable secreta previamente exportada, cabecera por stdin:

```bash
python -c 'import json,os; print("header = " + json.dumps("X-Service-API-Key: " + os.environ["ML_API_SERVICE_KEY"]))' | \
  curl --silent --show-error --fail-with-body --max-time 10 --config - \
  --header 'Content-Type: application/json' --header 'X-Request-ID: synthetic-explain-001' \
  --data-binary @examples/prediction_explain_rf.json \
  http://127.0.0.1:8001/internal/v1/predictions/explain

python -c 'import json,os; print("header = " + json.dumps("X-Service-API-Key: " + os.environ["ML_API_SERVICE_KEY"]))' | \
  curl --silent --show-error --fail-with-body --max-time 10 --config - \
  --header 'Content-Type: application/json' --header 'X-Request-ID: synthetic-batch-001' \
  --data-binary @examples/prediction_batch_compare.json \
  http://127.0.0.1:8001/internal/v1/predictions/batch
```

Para XGB/single cambiar únicamente el archivo a prediction_explain_xgb.json
o prediction_batch_single.json. Sin curl -v, secretos en argv ni volcados
del entorno. El presupuesto curl es de ejemplo local, no SLA.

```bash
python -m pytest -q
python -m pytest tests/api -q
ML_RUN_REAL_INTEGRATION=1 python -m pytest -q
ML_RUN_REAL_INTEGRATION=1 python -m pytest tests/api -q
python -m pip check
python scripts/export_ml_contract.py
python scripts/smoke_ml_api.py --extensions --port 8002 \
  --reference "$ML_MODEL_REFERENCE_DIR" --manifest-sha256 "$ML_MODEL_MANIFEST_SHA256" \
  --output reports/api-extension-new/smoke.json
python scripts/benchmark_ml_api.py --extensions --requests 5 \
  --reference "$ML_MODEL_REFERENCE_DIR" --manifest-sha256 "$ML_MODEL_MANIFEST_SHA256" \
  --output reports/api-extension-new/benchmark.json
```

Sin opt-in, seis casos reales se omiten explícitamente. Con artefactos y opt-in,
errores SHAP fallan como tests, nunca se convierten en skip.
El smoke verifica puerto libre, usa 8002 alterno, prueba las siete rutas y
tres 401 y apaga únicamente su proceso. Un sandbox puede pedir autorización
para bind. Los outputs deben ser nuevos, en reports ignorado.

| Negativo | Reproducción segura con TestClient |
| --- | --- |
| 401 | sin key/incorrecta/duplicada, incluso JSON roto |
| 409 | feature_version sintética incompatible |
| 422 | top_k bool/0/11, batch vacío/33, compare model_key, item inválido |
| 413 | supera ML_MAX_REQUEST_BYTES, con/sin Content-Length |
| 429 | cupos globales ocupados o segunda explicación simultánea |
| 503 EXPLANATION_NOT_READY | flag off o import/init opcional fallido |
| 503 MODEL_NOT_READY | demo desactivada/modelos no verificados |

`python -m pytest tests/api/test_extensions.py tests/api/test_security.py -q`
reproduce los controles sin datos de clientes ni cambios del servicio vivo.

Un worker, dos cupos globales y **una explicación simultánea** como máximo.
La segunda explicación recibe 429 y no ocupa indefinidamente otro cupo.
Locks por modelo: explain RF puede demorar otra predicción RF; XGB puede usar
el cupo restante. Una fila densificada con presupuesto 1 MiB (144 columnas
originales). Timeout HTTP no cancela cómputo iniciado; aislamiento de procesos
y capacidad de despliegue futuros, sin promesa de cancelación.

Benchmark TestClient en proceso, sin red/Fastify/UI: single RF/XGB, compare,
explain RF/XGB, batch 1/8/32 en ambos modos, concurrencia 2, p50/p95 y pico RSS.
Warm-up: 3 single y 1 por extensión. Cinco muestras artificiales no prueban
SLA ni generalización. Mediciones y warnings:
[informe nuevo](api_verification.md).

Dependencia opcional en un entorno nuevo: después de verificar resolución y
compatibilidad con el manifiesto, `python -m pip install -e '.[dev,ml,api,explain]'`.
En el entorno ya verificado de esta entrega no se ejecutó esa instalación:
`python -m pip check` pasó y no se alteró la resolución ML. Consultar la versión
con `python -c 'import importlib.metadata as m; print(m.version("shap"))'`.
