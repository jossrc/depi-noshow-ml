# Contrato futuro Fastify → Python

Documento de integración; no se instala ni implementa Node/Fastify/Next.js en este repositorio.

Flujo: `GetReservationRiskRoute → GetReservationRiskHandler → GetReservationRisk → ReservationRepository + RiskFeaturesBuilder → PredictionClient → FastAPI → PredictionResultRepository → respuesta web`.

Fastify es responsable de autenticación/roles, reservas, sedes autorizadas, cálculo de las 19 features desde snapshots legítimos al reservar, importaciones CSV/Excel, reportes, campañas, historial y eventual persistencia de scores/outcomes. Mantiene IDs de cliente/cita y secretos fuera de la petición Python. La existencia de una nueva clínica se valida en Fastify; Python tolera categorías no vistas a través del encoder guardado, sin fit. Esto no acredita desempeño sobre esa categoría.

El historial debe cumplir conocimiento anterior al corte (`known_at < prediction_at`) y las fuentes SCHEDULED deben representar el snapshot de reserva. No usar la tabla de entrenamiento, `client_history_current`, saldo actual de bono, resultados/tickets/laser finales ni catálogo posterior para simular conocimiento pasado. La declaración `snapshot_type` no certifica por sí sola la reconstrucción point-in-time. HU-04, score a 24 h, requiere diseño/entrenamiento temporal distinto; no alterar anticipación original. HU-14 trata candidato no-show, no cancelaciones generales. Registrar outcomes alimenta evaluación/reentrenamiento **offline**, nunca aprendizaje por request.

Tipos ilustrativos del contrato implementado (generar y validar a partir de OpenAPI para el release real):

```typescript
type MlModelKey = "random_forest_full_v1" | "xgboost_full_v1";
type FeatureVersion = "BOOKING_BASELINE_V1";
type Features = {
  age_at_booking: number | null; client_sex: "M" | "F" | "X" | null;
  booking_lead_days: number; appointment_month: number;
  appointment_weekday: number; appointment_hour: number;
  duration_minutes: number | null; is_fwa: boolean | null; clinic_id: number;
  scheduled_service_lines: number; distinct_body_areas: number;
  single_body_area_id: number | null; has_medical_evaluation: boolean | null;
  has_type4_service: boolean | null; previous_attended: number; previous_no_show: number;
  previous_no_show_rate: number | null; days_since_previous_attended: number | null;
  days_since_previous_no_show: number | null;
};
type CompareRequest = {
  feature_version: FeatureVersion;
  feature_context: { snapshot_type: "booking_creation"; builder_version?: string | null };
  features: Features;
};
type PredictionRequest = CompareRequest & { model_key?: MlModelKey | null };
type ModelScore = {
  api_version: "v1"; model_key: MlModelKey; algorithm: "random_forest" | "xgboost";
  model_version: string; model_hash: string; feature_version: FeatureVersion;
  probability_no_show: number; prediction_target: "NO_SHOW_CANDIDATE"; target_class: 1;
  risk_level: null; risk_policy_status: "pending_validation";
  validation_status: "EXPLORATORY_NOT_VALIDATED";
  calibration_status: "not_operationally_validated"; operational_use_allowed: false;
  request_id: string;
};
type CompareResponse = {
  api_version: "v1"; feature_version: FeatureVersion; request_id: string;
  results: [ModelScore, ModelScore];
  comparison: {
    absolute_probability_difference: number;
    interpretation: "descriptive_only_not_model_confidence";
    ensemble_probability: null; selected_winner: null;
  };
  risk_level: null; risk_policy_status: "pending_validation";
  validation_status: "EXPLORATORY_NOT_VALIDATED";
  calibration_status: "not_operationally_validated"; operational_use_allowed: false;
};
type MlError = {
  error: { code: string; message: string; details: { field: string; reason: string }[] };
  request_id: string;
};
```

TypeScript `number` no verifica enteros, finitud ni rangos; usar validación de runtime desde schema. Para IDs categóricos BIGINT, Node debe verificar `Number.isSafeInteger` o acordar un contrato nuevo si exceden su precisión. No enviar strings esperando coerción. Estas identidades/estados reflejan el MVP actual; una aprobación científica real requeriría un release de contrato/modelo y estados derivados de artefactos aceptados, sin editar este literal para simular promoción.

Ejemplo de cliente documental (no backend implementado):

```typescript
async function compareSyntheticSnapshot(body: CompareRequest): Promise<CompareResponse> {
  const response = await fetch(`${process.env.ML_SERVICE_URL}/internal/v1/predictions/compare`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Service-API-Key": process.env.ML_SERVICE_KEY!,
      "X-Request-ID": crypto.randomUUID(),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(5000), // presupuesto de ejemplo, por medir extremo a extremo
  });
  if (!response.ok) throw new Error(`ML unavailable: HTTP ${response.status}`);
  const data: unknown = await response.json();
  return validateCompareResponseFromOpenApi(data); // pendiente: runtime validator real
}
```

La función `validateCompareResponseFromOpenApi` es una responsabilidad pendiente explícita. No afirmar que este bloque es un cliente ejecutable completo. El timeout no cancela el cómputo ya admitido por Python. Acordar presupuesto, reintentos limitados y circuit breaker cuando exista backend.

| Resultado Python | Tratamiento Fastify/UI |
| --- | --- |
| 200 | Mantener modelo/algoritmo/versiones; mostrar estimaciones experimentales no calibradas |
| 401 | Corregir credencial interna; no reenviar clave al navegador |
| 404 UNKNOWN_MODEL | Corregir configuración; no escoger otro modelo silenciosamente |
| 409 | Corregir contrato, sin convertir features o inventar datos |
| 422 | Corregir RiskFeaturesBuilder, sin conservar payloads/PII en logs |
| 413 / 429 | Acotar tamaño/carga; no reintentar sin límite |
| 503 / 500 / timeout | Predicción no disponible; nunca 0% o riesgo bajo inventado |

En demo, una eventual persistencia autorizada deberá guardar timestamp, snapshot/origen/builder, versión de features, `model_key`, `model_version`, hash y estado por **cada** algoritmo. El hash corto HTTP identifica la referencia junto con su clave/versionado; conservar el SHA completo desde el catálogo privado del release. Historial no se recalcula ni reescribe silenciosamente. Idempotencia y retención son responsabilidad del backend.

Comparación de UI: mostrar RF y XGB con sus etiquetas, diferencia en puntos porcentuales **descriptiva**, semáforo no disponible. No promediar, elegir score mayor, llamar confianza a su cercanía, explicar con importancia global como causa local ni activar sobreasignaciones/campañas reales.

Pruebas futuras cuando exista Node: builder as-of, autenticación, JSON/schema, paridad Python, errores/timeouts, persistencia versionada sin duplicados, ausencia de secretos en web/logs y latencia completa Next.js→Fastify→Python. Esta entrega solo verifica Python y loopback curl, no E2E Node.
