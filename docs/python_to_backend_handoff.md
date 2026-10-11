# Handoff del servicio Python al backend

Verificación: 11-10-2026. **PYTHON_ML_READY_LOCAL=true** para inferencia técnica
exploratoria con los dos pipelines originales. No equivale a aprobación
científica ni a servicio comercial listo. El smoke inició y terminó su propio
Uvicorn; para dejarlo activo se debe ejecutar el comando de arranque.

## Arranque local

Desde `depi-noshow-ml`, con el entorno existente compatible:

```bash
source .venv/bin/activate
mkdir -p reports/ml-api-runtime/tmp reports/ml-api-runtime/mpl
export TMPDIR="$PWD/reports/ml-api-runtime/tmp"
export MPLCONFIGDIR="$PWD/reports/ml-api-runtime/mpl"
export ML_API_ENV=development
export ML_API_MODE=exploratory_demo
export ML_ALLOW_EXPERIMENTAL_INFERENCE=true
export ML_MODEL_REFERENCE_DIR=experiments/phase2-exploratory-prompt5-20261008
export ML_MODEL_MANIFEST_SHA256=ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac
export ML_DEFAULT_MODEL_KEY=random_forest_full_v1
export ML_ENABLE_LOCAL_EXPLANATIONS=true
export ML_MAX_BATCH_ITEMS=32
export ML_MAX_CONCURRENT_PREDICTIONS=2
export ML_EXPOSE_DOCS=true
export ML_API_SERVICE_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
python -m uvicorn depi_ml.api.main:create_app --factory --host 127.0.0.1 --port 8001 \
  --no-access-log --log-config examples/uvicorn_logging.json
```

El pin proviene del runbook versionado del original local, verificado otra vez;
no calcular un pin nuevo para aceptar un archivo desconocido. ApiSettings usa
solo variables del proceso y no carga `.env`. No necesita contraseña PostgreSQL.
Clave interna solo en `X-Service-API-Key`; nunca frontend, URL, logs o argumentos
literales. Distribuirla por gestor de secretos/entrada oculta. No publicar demo
en staging/production: el modo actual falla cerrado fuera de development.

## Contrato y rutas

URL: `http://127.0.0.1:8001`. OpenAPI y tipos de referencia están en
[docs/contracts](contracts/ml-internal-v1.openapi.json); se comprobó que la
regeneración actual produce exactamente los mismos bytes.

| Método | Ruta | Uso |
| --- | --- | --- |
| GET | /health/live | Vida del proceso, sin autenticación |
| GET | /health/ready | Ambos pipelines disponibles; 200 o 503 |
| GET | /internal/v1/models | Catálogo autenticado |
| POST | /internal/v1/predictions | RF o XGB individual |
| POST | /internal/v1/predictions/compare | Dos scores independientes |
| POST | /internal/v1/predictions/explain | Predicción + SHAP local opcional |
| POST | /internal/v1/predictions/batch | Single/compare con items ordenados |

Las rutas internas exigen `X-Service-API-Key`. `X-Request-ID` opaco es opcional,
máximo 64 caracteres; no incluir identificadores personales. No hay endpoints
de búsqueda de reservas, escritura de scores, entrenamiento ni sincronización.

Modelos: `random_forest_full_v1:4e3c86a17005` y
`xgboost_full_v1:376d3db9d4d3`. Versión `BOOKING_BASELINE_V1`, target
`NO_SHOW_CANDIDATE`, clase 1. El default RF es provisional, sin ganador/ensemble.

Las 19 claves, en orden canónico, son:

1. age_at_booking
2. client_sex
3. booking_lead_days
4. appointment_month
5. appointment_weekday
6. appointment_hour
7. duration_minutes
8. is_fwa
9. clinic_id
10. scheduled_service_lines
11. distinct_body_areas
12. single_body_area_id
13. has_medical_evaluation
14. has_type4_service
15. previous_attended
16. previous_no_show
17. previous_no_show_rate
18. days_since_previous_attended
19. days_since_previous_no_show

Tipos/nulls/rangos exactos: [api_ml.md](api_ml.md) y OpenAPI. Son obligatorias
las 19 claves, incluso las nullables; JSON estricto sin extras ni strings como
números. Booleanos son true/false. Sin historial, rate/recencias son null según
contrato; con historial, rate debe ser el cociente de contadores. No rellenar
ausencias arbitrariamente. IDs personales, target y cancelaciones no son inputs.

Raíz exige `feature_version` y
`feature_context.snapshot_type="booking_creation"`. El backend debe demostrar
que ese snapshot existió al reservar; etiquetar datos actuales como creación
no cumple el contrato. `builder_version` es metadato opcional, no predictor.

Ejemplos completos públicos y sintéticos, sin personas ni credenciales:
[single](../examples/prediction_single.json),
[compare](../examples/prediction_compare.json),
[SHAP RF](../examples/prediction_explain_rf.json),
[SHAP XGB](../examples/prediction_explain_xgb.json),
[batch single](../examples/prediction_batch_single.json) y
[batch compare](../examples/prediction_batch_compare.json).
No usar el CSV histórico para inventar un snapshot de una reserva futura.

## Interpretación y errores

`validation_status=EXPLORATORY_NOT_VALIDATED`,
`calibration_status=not_operationally_validated`, `operational_use_allowed=false`,
`risk_level=null`, `risk_policy_status=pending_validation`. Conservar estos
estados en persistencia/interfaz; no convertir score o umbral 0.5 en semáforo.
La diferencia RF/XGB es descriptiva, no confianza ni consenso.

SHAP RF explica probabilidad; XGB explica log-odds con enlace sigmoid.
`base_value + factors + other_features_contribution` reconstruye la salida en
ese espacio. Los factores son asociaciones locales, sin causalidad ni valores
de entrada. `top_k` 1–10 solo limita presentación. No comparar magnitudes entre
espacios. Una explicación opcional no disponible no bloquea single/compare.

| HTTP | Condición / acción del consumidor |
| --- | --- |
| 400 | JSON/content type inválido; corregir transporte |
| 401 | Clave ausente, incorrecta o duplicada; revisar credencial |
| 404 | Model key desconocida o ruta inexistente |
| 409 | Feature version incompatible; corregir contrato |
| 413 | Payload excede 262144 bytes por defecto; fragmentar |
| 422 | Tipos/rangos/extras/contexto/top_k/lote inválidos |
| 429 | Cupos por worker ocupados; backpressure y reintento acotado |
| 500 | INFERENCE_FAILED o EXPLANATION_FAILED, sin score parcial |
| 503 | MODEL_NOT_READY o EXPLANATION_NOT_READY; no sustituir por cero |

Batch: 1–32 items por defecto, máximo configurado 64; contexto común,
compare prohíbe model_key. Validación atómica antes de inferencia. Respuesta
ordenada con índices; error intermedio devuelve error, sin persistencia parcial.
Un worker tiene dos cupos y una explicación simultánea. HTTP timeout no cancela
el thread ya iniciado. Medir extremo a extremo; p95 local no constituye SLA.

## Trabajo requerido en Node/ETL, aún no implementado

- Construir las 19 features con evidencia de disponibilidad al crear reserva,
  servicios originales por copia/revisión, catálogo histórico y eventos de
  historial conocidos estrictamente antes del corte. No usar historial actual,
  balances presentes, target futuro ni características de atención/facturación.
- Auditar 24 enlaces rotos, 114 detalles huérfanos y vigencia del flag actual
  training_eligible. Un par con detalles conservados no acredita su inmutabilidad.
- Resolver con el responsable del ETL la convención de domingo: el original
  observado solo tiene valores 1–6, iguales para DOW/ISODOW; HTTP exige ISO 1–7
  mientras auditoría offline histórica usa DOW 0–6. No se probó cómo se generaría
  domingo. No cambiar semántica ni mapear 0→7 sin revisión/versionado explícitos.
- Mapear exactamente claves, tipos/nulls y reglas; validar contra OpenAPI antes
  de llamar a Python. Guardar snapshot, corte, builder, hashes/model versions,
  states y request ID mediante una implementación futura autorizada del backend.
- Implementar autenticación de usuarios/RBAC, cliente interno, idempotencia,
  invalidación ante cambios de reserva, manejo de errores y endpoints de interfaz.
  Definir esquema/persistencia de scores en ese repositorio, sin asumir objetos
  nuevos ni escrituras desde FastAPI.
- Mantener estado exploratorio y riesgo null. Solicitar SHAP bajo demanda,
  fragmentar batch y medir timeout/backpressure/memoria del despliegue real.

Python no consulta PostgreSQL durante inferencia, no verifica copias por sí
solo y no obtiene datos externos. Node/Next no se abrieron ni modificaron.
Refresco futuro: [future_data_refresh_runbook.md](future_data_refresh_runbook.md).
