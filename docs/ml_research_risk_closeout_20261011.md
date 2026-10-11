# Cierre técnico de riesgo experimental — 11-10-2026

**research_risk_implemented=true; risk_bands_ready_local=true;
operational_risk_approved=false; EXPLORATORY_NOT_VALIDATED.**

Este es un encargo posterior a la recuperación Python. Se implementó una
superficie v2 aditiva; v1 conserva exactamente probabilidades, 19 entradas,
schemas, semántica de riesgo null, SHAP, lotes y estados exploratorios.
Base limpia inspeccionada: `main`, `b6a7133474564f54d0f1e900045d29d2870b2314`.
No se hicieron commits ni push, ni se accedió a código Node/Next.

## COMPLETED

Políticas activas por modelo: `research_risk_v1_1`, publicación técnica provisional
en [policies/research_risk_v1_1](../policies/research_risk_v1_1/publication.json).
No implica aprobación científica o comercial. Umbrales calculados con los
scores originales de 33.808 reservas de abril–mayo, retenidas por las reglas
originales de madurez con corte exclusivo 1 de julio 2026. Método fijado antes
de observar los límites: cuantiles 1/3 y 2/3, interpolación linear, 12 decimales
publicados. No se redondea ni recalibra la probabilidad servida.

| Modelo | LOW si p < | HIGH si p >= | MEDIUM |
| --- | ---: | ---: | --- |
| RF full original | 0.302192334412 | 0.460052607619 | Intervalo entre límites, inclusivo por izquierda |
| XGB full original | 0.243454828858 | 0.435159325600 | Intervalo entre límites, inclusivo por izquierda |

Implementación inmutable de política, fuente/evidencia tipadas, hashes canónicos,
registro pineado compatible con ambos modelos y clasificador puro compartido
por análisis/serving. Políticas inválidas/faltantes/mismatch deshabilitan solo
v2; v1 sigue disponible. No hay acceso a CSV, SQL, joblib o archivos por request.

Nuevas rutas autenticadas:

- GET `/internal/v2/risk-policy`.
- POST `/internal/v2/predictions`.
- POST `/internal/v2/predictions/compare`.
- GET `/internal/v2/openapi.json`, solo con docs habilitadas en development.

Requests de predicción son los mismos schemas v1: las mismas 19 claves,
`BOOKING_BASELINE_V1`, `NO_SHOW_CANDIDATE`, clase 1 y `booking_creation`.
Responses añaden porcentaje, hash completo del modelo y `experimental_risk`
con nivel, etiqueta exploratoria, límites, versión/hash, semántica y advertencias.
No se añadió v2 batch/explain: los existentes v1 se conservan y fueron probados
con research activado. No se implementó un segundo motor de inferencia.

Flag nuevo `ML_ENABLE_RESEARCH_RISK=false` por defecto, estricto true/false,
además de las barreras originales de demo development, key y modelos confiables.
Path y pin explícitos; no se autopinea una política durante startup.

Archivos nuevos principales: domain/research_risk_policy.py,
infrastructure/ml/research_risk_registry.py, analysis/research_risk.py,
api/schemas/research_risk.py, api/routes/research_predictions.py,
api/openapi_versions.py; scripts build/publish/export/benchmark research;
tests unit/API/reales; ejemplos sintéticos; contratos v2 y políticas agregadas.
Cambios existentes: api/main.py, ApiSettings, smoke_ml_api.py, .env.example
y adiciones explícitas de documentación. Detalle en summary.json y git diff.

## METRICS

Validación original reproducida: 41.427 filas de abril–junio, 9.924 positivos,
prevalencia 23,955391%; umbral diagnóstico fijo 0.5. No se retocaron cortes.

| Modelo | ROC-AUC | AP | Precision | Recall | F1 | Brier | Log-loss | ECE 10 bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 0.750229 | 0.512569 | 0.482153 | 0.509069 | 0.495246 | 0.175586 | 0.527998 | 0.144360 |
| XGB | 0.724693 | 0.473073 | 0.484682 | 0.417674 | 0.448690 | 0.171722 | 0.510490 | 0.100344 |

Matrices: RF `[[26077,5426],[4872,5052]]`; XGB
`[[27096,4407],[5779,4145]]`. AP es average precision.

Diagnóstico junio: **7.619 filas, 1.055 positivos**, con labels elegibles
anteriores al corte original de 1 de julio. Umbrales fijados solo con scores
abril–mayo. Junio fue expuesto a validación/modelos/hiperparámetros previos:
**no es evaluación independiente del modelo**.

| Modelo | Banda | N junio | Positivos | Tasa % | IC95 % por cliente |
| --- | --- | ---: | ---: | ---: | --- |
| RF | LOW | 2841 | 133 | 4.681 | 3.923–5.482 |
| RF | MEDIUM | 2511 | 272 | 10.832 | 9.704–12.051 |
| RF | HIGH | 2267 | 650 | 28.672 | 26.672–30.598 |
| XGB | LOW | 3047 | 189 | 6.203 | 5.392–7.047 |
| XGB | MEDIUM | 2570 | 329 | 12.802 | 11.514–14.112 |
| XGB | HIGH | 2002 | 537 | 26.823 | 24.812–28.775 |

Desarrollo: RF N=11270/11269/11269, tasas
10.089%/21.910%/46.703%; XGB N=11269/11271/11268,
tasas 11.261%/23.547%/43.894%, en LOW/MEDIUM/HIGH.
Los conteos de terciles no son forzosamente iguales: fronteras/empates quedan
fijos y en junio cambia la cobertura. La tasa dentro de cada banda cae en junio;
no representa una probabilidad absoluta estable.

IC95 bootstrap percentil por cliente: 1000 réplicas, seed 42, condicionados
a modelo y límites fijos. No capturan incertidumbre de ajuste/selección,
errores históricos, shocks de período o cambios de negocio. Sedes requieren
>=30 filas por celda y >=90 por cohorte/sede; IC requiere >=20 clientes.
Se encontraron 120 grupos completos de sede/período y 2 sin monotonicidad
descriptiva; se reportan sin ocultarlos ni ajustar límites por sede.

Candidatos de calibración **offline, no servidos**: sigmoid/logit con C=1e6 e
isotónica ajustados en 28.341 filas abril–mayo con labels registrados antes
de 1 de junio, disjuntas de train. Evaluación solo junio. Brier junio
RF raw/sigmoid/isotónica: 0.161818/0.118630/0.117944;
XGB: 0.148573/0.120979/0.120292. No se serializaron/promovieron calibradores.
No se llamó a fit de los RF/XGB originales. [Separación de datos de calibración](https://scikit-learn.org/stable/modules/calibration.html).

## PROOF

- Baseline anterior: 490 pruebas y 23 comprobaciones HTTP v1.
- Release final: **558 passed, 0 failed, 0 skipped** con integración real;
  default: **550 passed, 8 skipped** opt-in explícito, sin fallos.
- **45 checks HTTP reales** con v1 y research v1_1, incluidos cuatro fixtures
  que cubren LOW/MEDIUM/HIGH en ambos modelos, y negativos 400/401/409/413/422.
  429, fallos 503 de policy/mismatch y modos staging/production se verificaron
  mediante escenarios controlados de suite.
- Rollback por HTTP real: 25 checks, catálogo v2 UNAVAILABLE/disabled y
  predicción 503 mientras v1 ready/single/compare/batch/SHAP siguen respondiendo.
- V1 OpenAPI y tipos originales idénticos; modelos/datasets/auxiliares/manifests
  originales preservados. Políticas v1_1 generadas dos veces con bytes idénticos.
- pip check, py_compile y git diff --check pasaron. Warning existente de
  Starlette/httpx documentado; no se actualizaron dependencias.
- No PostgreSQL requerido en esta ejecución offline; **0 escrituras PG,
  0 mutaciones Docker**, sin FLOWww sync y test sin scores/evaluación.
  TMPDIR/MPLCONFIGDIR/basetemp dentro de la carpeta nueva del repo.

Pin del archivo de políticas v1_1:
`459b9b6da3f0a724df787606630942386928ddee8758439cb3da9131a30b075a`.
Policy hash RF:
`1d58ce962e899b758275f7191244225e8c970cf2db9661ffcb21be9b63a4ce74`;
XGB:
`208c706d839b7c8b71134cd225da4bc1bb935c5d4bd06cbb898a8d8f84ee7032`.
Pipelines: RF `4e3c86a17005…`, XGB `376d3db9d4d3…`; hashes completos
en artifact/publication y respuestas v2. Pin del experimento original sin cambios.

Corrección numérica auditada: la primera research_risk_v1 queda conservada y
sustituida. np.where con array float32 reducía a float32 la frontera decimal XGB;
dos negativos de desarrollo quedaban HIGH en vez de MEDIUM bajo la comparación
exacta servida. Se compartió una sola regla de dominio con conversión explícita
de score a float64. Se corrigieron evidencia/hash en v1_1 y se añadió regresión.
**Umbrales, scores y todas las métricas de junio permanecen iguales**. La versión
inicial ya no es aceptada por el loader final; no se sobrescribió su JSON.

Benchmark local TestClient, 20 muestras/escenario, un worker y concurrencia 2:
p95 catálogo 0.79 ms; RF v2 33.85 ms; XGB v2 7.23 ms;
compare 39.67 ms, compare concurrente 73.87 ms. Startup 2.859 s;
pico RSS después 578.47 MiB, sin SHAP habilitado en ese benchmark.
No es SLA, throughput sostenido o medición Node/FE.

## NOT_APPROVED / LIMITATIONS

LOW/MEDIUM/HIGH son **segmentos relativos del score crudo** para investigación,
no intervalos de riesgo absoluto, cancelación o certeza personal. El nivel HIGH
puede tener una tasa observada de 26–29% en junio. Todas las respuestas incluyen
RESEARCH_PROVISIONAL, advertencias, approved_by=null y operational_use_allowed=false.

Persisten snapshots históricos/catálogo sin acreditación completa, labels
registrados como proxy de disponibilidad, sesgo de elegibilidad/madurez y
clientes recurrentes entre períodos. Transición comercial junio–julio aproximada,
sin día exacto ni causalidad única demostrada; rendimiento posterior desconocido.
Calibración/umbrales comerciales, costos/capacidad, revisión humana y evaluación
independiente final no están aprobados. No se usó test reservado para seleccionar,
calibrar, ajustar o diagnosticar esta política. Ningún umbral fue inventado/retocado
para mejorar etiquetas de junio.

## NODE HANDOFF

[Contrato OpenAPI v2](contracts/ml-internal-v2.openapi.json),
[tipos derivados](contracts/ml-internal-v2.types.ts),
[handoff aditivo completo](python_to_backend_handoff.md#v2-research-risk),
[fixtures públicas](../examples/research_risk_synthetic_cases.json).
Node consume `experimental_risk.level/label` y persiste versión/hash/modelo;
no diseña cortes ni mezcla modelos. No usar riesgo legacy null como banda v2.
La clave interna no debe llegar al navegador. No tratar UNAVAILABLE como LOW.
Conserva advertencias y bloqueo de acciones automáticas por operational_use_allowed=false.

Python no obtiene reservas ni reconstruye snapshots reales. Node aún debe
acreditar booking_creation, servicios por versión e historial as-of, persistir
idempotentemente y medir cliente/timeout/backpressure/UI. Sin evidencia upstream
puede haber NOT_SCORABLE; no se falsea snapshot para obtener un badge.

Todos los checkpoints técnicos del nuevo encargo están respaldados por
[summary.json](../reports/risk-policy-closeout-20261011-165943/summary.json).
Evidencia agregada en `reports/risk-policy-closeout-20261011-165943/`:
risk_policy, risk_bands_metrics CSV/JSON, calibration_diagnostics,
model_comparison, policy_review, HTTP release/disabled, qa_results,
openapi_v2, hashes, JUnit, benchmark y handoff. Sin filas personales.
