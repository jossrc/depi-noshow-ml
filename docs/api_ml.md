# Contrato HTTP interno ML v1

Servicio para demo sintética local. Dos pipelines originales `full`, 19 predictoras, `BOOKING_BASELINE_V1`. No predice cancelaciones generales ni ofrece acciones operativas.

| Método | Ruta exacta | Autenticación | Respuesta |
| --- | --- | --- | --- |
| GET | `/health/live` | No, solo red interna | 200 proceso vivo |
| GET | `/health/ready` | No, solo red interna | 200 ambos modelos listos; 503 si falla cualquiera |
| GET | `/internal/v1/models` | `X-Service-API-Key` | 200 catálogo y contrato efectivo |
| POST | `/internal/v1/predictions` | `X-Service-API-Key` | 200 score del modelo solicitado/default |
| POST | `/internal/v1/predictions/compare` | `X-Service-API-Key` | 200 RF y XGB independientes, diferencia absoluta |

`X-Request-ID` opcional: 1–64 caracteres `[A-Za-z0-9._-]`, primero alfanumérico. Debe ser opaco, sin IDs personales. Valores ausentes, duplicados o inválidos se sustituyen por UUID hexadecimal. Se devuelve en cabecera, scores y errores. No hay CORS abierto ni rutas de negocio, entrenamiento o upload. PROMPT8 agrega explain y batch descritos abajo.

POST requiere `application/json` UTF-8. Request y contexto rechazan extras. Todas las 19 claves son obligatorias, incluso las que admiten null; no hay imputación HTTP. Booleanos JSON `true/false`, enteros estrictos, números finitos sin coerción desde strings, `NaN`/Infinity rechazados. JSON mal formado, claves duplicadas o codificación inválida producen 400.

El schema se deriva de `datasets/schema.py::PREDICTOR_COLUMNS`; OpenAPI detalla tipos, nulos y rangos. Políticas complementarias:

- Edad: 16–95 o null; sexo: M/F/X o null; anticipación >0; mes 1–12, ISODOW 1–7, hora 0–23.
- Duración ≥0 o null; líneas, áreas y contadores históricos ≥0; clínica y área única enteros positivos. No se restringen sedes a categorías observadas en 2026.
- Áreas ≤líneas; ID de área única presente exactamente cuando hay una única área, según agregación SQL documentada. Cero áreas con líneas programadas es admisible.
- Sin eventos históricos, tasa null. Con historial, tasa requerida e igual a no-show/total con tolerancia absoluta `1e-6`. No se acepta una tasa null con historial.
- Recencias finitas ≥0 o null; contador cero exige recencia null. Con contador positivo se conserva null como desconocido sin inventar días, por la nullabilidad declarada; upstream debe contrastar esta situación con su snapshot y la auditoría offline, que detectaba historia sin recencia. Esta flexibilidad HTTP no certifica calidad temporal.
- Booleanos operativos admiten null y se conservan. Valores ausentes no se sustituyen por cero.
- Solo `feature_context.snapshot_type="booking_creation"`. `builder_version` es opcional y acotada a 64 caracteres alfanuméricos, punto, guion o underscore. Es metadato, no predictor.
- `feature_version` es requerida. Una versión string válida pero incompatible produce **409**. Un contexto de 24 h produce 422.
- `model_key` en single, explain y batch single; opcional/null usa default. Una clave de formato correcto sin modelo produce **404**, sin sustitución automática.

Ejemplo **artificial completo**, conservado también en `examples/prediction_compare.json`:

```json
{
  "feature_version": "BOOKING_BASELINE_V1",
  "feature_context": {"snapshot_type": "booking_creation", "builder_version": "synthetic_builder_v1"},
  "features": {
    "age_at_booking": 28, "client_sex": "F", "booking_lead_days": 9.5,
    "appointment_month": 11, "appointment_weekday": 2, "appointment_hour": 15,
    "duration_minutes": 40, "is_fwa": false, "clinic_id": 1,
    "scheduled_service_lines": 1, "distinct_body_areas": 1, "single_body_area_id": 101,
    "has_medical_evaluation": false, "has_type4_service": false,
    "previous_attended": 3, "previous_no_show": 1, "previous_no_show_rate": 0.25,
    "days_since_previous_attended": 30, "days_since_previous_no_show": 110
  }
}
```

`101` es un ID inventado para verificar el formato; no representa un código de tratamiento identificado. Para predicción individual añadir `"model_key":"random_forest_full_v1"` o `"xgboost_full_v1"`, como en `examples/prediction_single.json`.

Respuesta individual observada con RF y la fixture artificial (request ID ilustrativo):

```json
{
  "api_version": "v1",
  "model_key": "random_forest_full_v1",
  "algorithm": "random_forest",
  "model_version": "random_forest_full_v1:4e3c86a17005",
  "model_hash": "4e3c86a17005",
  "feature_version": "BOOKING_BASELINE_V1",
  "probability_no_show": 0.6976400433344135,
  "prediction_target": "NO_SHOW_CANDIDATE",
  "target_class": 1,
  "risk_level": null,
  "risk_policy_status": "pending_validation",
  "validation_status": "EXPLORATORY_NOT_VALIDATED",
  "calibration_status": "not_operationally_validated",
  "operational_use_allowed": false,
  "request_id": "synthetic-request-001"
}
```

Comparación: `results` tiene exactamente dos objetos con **ese mismo esquema completo**, primero RF y luego XGB. Para esta fixture se observaron RF `0.6976400433344135` y XGB `0.9159286618232727`. Además devuelve:

```json
{
  "absolute_probability_difference": 0.21828861848885917,
  "interpretation": "descriptive_only_not_model_confidence",
  "ensemble_probability": null,
  "selected_winner": null
}
```

Ese objeto está en `comparison`; en la raíz también aparecen `api_version`, `feature_version`, `request_id`, `risk_level=null`, `risk_policy_status=pending_validation`, `validation_status=EXPLORATORY_NOT_VALIDATED`, `calibration_status=not_operationally_validated` y `operational_use_allowed=false`. La respuesta completa obtenida con curl está en el reporte local `reports/api-prompt7-20261008/http_smoke.json`. Los números son ejecución de una entrada artificial, no métricas ni probabilidades calibradas sobre personas reales. No están hardcodeados en el servicio.

Catálogo: `default_model_key`, versión y dos entradas con identidad, hash corto, `model_version`, algoritmo, experimento, training mode, estado, 19 nombres en `predictor_columns`, `feature_count` y rol `provisional_demo_default`/`independent_comparator`. SHA completo y versiones del entrenamiento permanecen en manifiesto y logs de startup protegidos; no se devuelve filesystem ni identidad del dataset.

Errores uniformes, sin valores de entrada, contexto Pydantic, rutas, claves ni stack:

```json
{
  "error": {
    "code": "INVALID_FEATURES",
    "message": "Los datos de predicción no cumplen el contrato v1.",
    "details": [{"field": "features.booking_lead_days", "reason": "greater_than"}]
  },
  "request_id": "synthetic-request-001"
}
```

Máximo 20 detalles; nombres extra arbitrarios se reemplazan por `[extra]`. Validaciones cruzadas usan `reason=value_error`, sin serializar la excepción. Errores de transporte tienen details vacío.

| HTTP | Código | Condición |
| --- | --- | --- |
| 400 | INVALID_REQUEST | JSON/content type/Content-Length inválido |
| 401 | UNAUTHORIZED_SERVICE | Clave ausente, incorrecta, no ASCII o duplicada; antes de parsear |
| 404 | UNKNOWN_MODEL / NOT_FOUND | Modelo desconocido / ruta inexistente |
| 405 | METHOD_NOT_ALLOWED | Método HTTP incorrecto |
| 409 | FEATURE_VERSION_MISMATCH | Versión incompatible |
| 413 | PAYLOAD_TOO_LARGE | Supera bytes permitidos, incluido chunked |
| 422 | INVALID_FEATURES | Contrato/tipos/rangos/contexto/no finitos inválidos |
| 429 | RATE_LIMITED | Capacidad de inferencia por worker agotada |
| 500 | INFERENCE_FAILED | Fallo controlado de pipeline o inesperado; sin resultado parcial |
| 503 | MODEL_NOT_READY | Demo desactivada o cualquiera de los modelos no disponible |

No se ofrece 504 como si se pudieran cancelar threads ML. El consumidor configura un timeout de red; no tratar un timeout/503 como probabilidad cero. Autenticación, límite de bytes y validación son controles distintos; una llamada no autorizada se rechaza antes de procesar su cuerpo.

`/docs` y `/openapi.json` existen solamente en development con `ML_EXPOSE_DOCS=true`. Exposición solo loopback/red interna; Swagger no es interfaz de negocio. El esquema incluye APIKeyHeader y respuestas de error explícitas. Fuera de development o sin el flag, ambas rutas responden 404.

## Ampliación aditiva PROMPT8: explicación y lotes

Las cinco rutas anteriores y sus schemas se contrastan con
`tests/api/fixtures/prompt7_openapi.json`. Solo se añaden:

| Método/ruta | Request | Response 200 | Auth |
| --- | --- | --- | --- |
| POST /internal/v1/predictions/explain | ExplanationRequest: PredictionRequest completo + top_k | ExplanationResponse: PredictionResponse completo + explanation | X-Service-API-Key |
| POST /internal/v1/predictions/batch | BatchSingleRequest o BatchCompareRequest, discriminador mode | BatchSingleResponse o BatchCompareResponse, discriminador mode | X-Service-API-Key |

Errores comunes 400/401/404/409/413/422/429/500/503 conservados.
Explain agrega 503 `EXPLANATION_NOT_READY` y 500 `EXPLANATION_FAILED` con el
mismo envelope. Fallo de predicción dentro de explain conserva
`INFERENCE_FAILED`. Modelo desconocido: 404 incluso con SHAP off.
Registry fallido: `MODEL_NOT_READY`. El catálogo `/models` se conserva sin
añadir campos de capacidades; disponibilidad opcional documentada en OpenAPI
y reflejada por el resultado de explain, nunca asumida.

Las 19 claves obligatorias conservan esta tabla:

| Feature | Tipo JSON / rango | null |
| --- | --- | --- |
| age_at_booking | entero 16–95 | sí |
| client_sex | M/F/X | sí |
| booking_lead_days | finito >0 | no |
| appointment_month | entero 1–12 | no |
| appointment_weekday | entero ISO 1–7 | no |
| appointment_hour | entero 0–23 | no |
| duration_minutes | entero ≥0 | sí |
| is_fwa | booleano estricto | sí |
| clinic_id | entero >0 | no |
| scheduled_service_lines | entero ≥0 | no |
| distinct_body_areas | entero ≥0, ≤líneas | no |
| single_body_area_id | entero >0; presente iff áreas=1 | sí |
| has_medical_evaluation | booleano estricto | sí |
| has_type4_service | booleano estricto | sí |
| previous_attended | entero ≥0 | no |
| previous_no_show | entero ≥0 | no |
| previous_no_show_rate | número [0,1], ratio histórico ±1e-6 | sí solo si total=0 |
| days_since_previous_attended | finito ≥0; null si contador=0 | sí |
| days_since_previous_no_show | finito ≥0; null si contador=0 | sí |

Con contador positivo la recencia desconocida admite null como antes.
Cada item batch tiene solo `features`; contexto/versión compartidos en raíz.
Sin PII, IDs de cita/cliente, targets ni política comercial como nueva feature.

### Explicación local

`top_k` entero estricto 1–10, default 5; no bool/float/string.
`model_key` opcional/null conserva default RF. Ejemplos completos:
[RF](../examples/prediction_explain_rf.json),
[XGB](../examples/prediction_explain_xgb.json). No se devuelven valores de entrada.

| Campo explanation | Semántica |
| --- | --- |
| method / feature_perturbation | tree_shap / tree_path_dependent |
| output_space / output_link | RF probability/identity; XGB log_odds/sigmoid, unión discriminada |
| base_value | salida esperada Tree SHAP en el espacio declarado |
| model_output_in_explanation_space | probabilidad RF o margen original XGB |
| reconstructed_output_in_explanation_space | base + suma de las 19 contribuciones |
| additivity_error | diferencia absoluta reconstrucción/salida |
| explained_class / total_feature_count | 1 / 19 |
| top_k / factors | hasta k variables agrupadas, ordenadas por magnitud |
| other_features_contribution | suma **con signo** de variables omitidas |
| explanation_scope | local_association_not_causality |

Cada factor contiene `feature_name` canónico, `contribution`,
`absolute_contribution` y `direction`: increases_model_output,
decreases_model_output o neutral para cero exacto. Empates por orden
`PREDICTOR_COLUMNS`; contribuciones pequeñas no se redondean a cero.
`top_k` limita presentación; aditividad siempre usa todos los factores.

Verificado SHAP 0.52.0: RF `(1,144,2)` y XGB `(1,144)`.
Clase 1 seleccionada por `classes_`; lista histórica RF de dos clases admitida,
RF `(1,p)` ambiguo rechazado. Agrupación por `transformers_`,
`output_indices_`, `categories_` y `get_feature_names_out`, sin partir nombres
por underscore. Cada columna transformada asignada una vez y las 19 originales
presentes. Topología drop/infrequent distinta de los originales se rechaza.

Solo `transform` del FeatureContract y ColumnTransformer ajustados.
RF densifica una fila preservando ceros. XGB transforma entradas **ausentes**
CSR en NaN y preserva ceros **almacenados**: un toarray de ceros cambiaría
la semántica del modelo. Sin background artificial ni lectura de CSV.
Tree path dependent utiliza la estructura ajustada del árbol:
[TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

Tolerancias: aditividad RF `1e-7`, XGB margen `1e-5`; enlace del margen
original a predict_proba `1e-7`. Integración comprueba sigmoid de reconstrucción
con `1e-6` y paridad de scores con `1e-10` absolutos: RF paralelo puede variar
una unidad de último lugar. SHAP representa asociaciones del modelo,
**no causas reales**. Log-odds no son puntos porcentuales.

### Lotes acotados

`mode` requerido: single/compare. Single permite `model_key` opcional/null;
compare **prohíbe** esa propiedad, incluso null. `items` array requerido no
vacío, ≤32 por defecto; `ML_MAX_BATCH_ITEMS` configurable 1–64. OpenAPI expresa
máximo absoluto 64; el límite configurado se comprueba antes del motor.
Exceso: 422 con safe field items.

Ejemplos [single](../examples/prediction_batch_single.json) y
[compare](../examples/prediction_batch_compare.json).
Raíz: api_version, feature_version, mode, count, results, request_id,
risk_level=null, risk_policy_status=pending_validation, validation_status,
calibration_status y operational_use_allowed=false.
Resultados: `index` consecutivo desde cero y `prediction`, un
**PredictionResponse completo** o **CompareResponse completo**.
RF→XGB en compare, sin ensemble/ganador. Cabecera, raíz e internos comparten ID.

Validación all-or-nothing: item inválido rechaza todo el request con **cero**
llamadas a predictores. Error seguro `items[3].features.age_at_booking`;
nombres extra se reemplazan por `[extra]`. Fallo de inferencia intermedio:
500 sin salida parcial; algunas filas pudieron computarse, sin persistencia.

Un lote ocupa un cupo global durante toda la ejecución. Filas ordenadas
reutilizan `_predict` y locks existentes: se eligió esta ruta para conservar
paridad sin introducir un segundo camino vectorizado. Sin thread/proceso
por item ni colas. Fastify fragmentará importaciones y aplicará backpressure.
Límite de bytes global sigue aplicando al cuerpo completo, incluido chunked.

### Contrato publicado

[OpenAPI](contracts/ml-internal-v1.openapi.json) y
[TypeScript derivado](contracts/ml-internal-v1.types.ts) regenerables con
`python scripts/export_ml_contract.py` sin lifespan/modelos/secretos.
Tests de snapshot, compatibilidad PROMPT7 y cuatro ejemplos Swagger válidos.
Sin rutas de negocio, upload, train/evaluate, activate, CRUD o metrics.
