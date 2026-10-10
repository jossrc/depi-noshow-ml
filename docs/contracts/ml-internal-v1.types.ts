// Generated from ml-internal-v1.openapi.json by scripts/export_ml_contract.py.
// Reference only; no Fastify client or runtime validation is implemented here.
// Validate JSON against OpenAPI: no extras, finite numbers, explicit required
// nulls, integer/range constraints and cross-feature consistency. In JS use
// Number.isSafeInteger for integer fields and Number.isFinite for scores/floats.
// Header X-Service-API-Key belongs only in Fastify; never send it to a browser.
// Correlate opaque X-Request-ID (1..64 ASCII [A-Za-z0-9._-]); never personal IDs.
export type MlModelKey = "random_forest_full_v1" | "xgboost_full_v1";
export type FeatureVersion = "BOOKING_BASELINE_V1";
export type MlRequestHeaders = { "X-Request-ID"?: string; "X-Service-API-Key": string };
export type BatchCompareRequest = {
  "feature_context": FeatureContext;
  "feature_version": string;
  "items": Array<BatchItem>;
  "mode": "compare";
};

export type BatchCompareResponse = {
  "api_version"?: "v1";
  "calibration_status"?: "not_operationally_validated";
  "count": number;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "mode": "compare";
  "operational_use_allowed"?: false;
  "request_id": string;
  "results": Array<CompareBatchItemResult>;
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type BatchItem = {
  "features": PredictionFeatures;
};

export type BatchSingleRequest = {
  "feature_context": FeatureContext;
  "feature_version": string;
  "items": Array<BatchItem>;
  "mode": "single";
  "model_key"?: string | null;
};

export type BatchSingleResponse = {
  "api_version"?: "v1";
  "calibration_status"?: "not_operationally_validated";
  "count": number;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "mode": "single";
  "operational_use_allowed"?: false;
  "request_id": string;
  "results": Array<SingleBatchItemResult>;
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type CompareBatchItemResult = {
  "index": number;
  "prediction": CompareResponse;
};

export type CompareRequest = {
  "feature_context": FeatureContext;
  "feature_version": string;
  "features": PredictionFeatures;
};

export type CompareResponse = {
  "api_version"?: "v1";
  "calibration_status"?: "not_operationally_validated";
  "comparison": Comparison;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "operational_use_allowed"?: false;
  "request_id": string;
  "results": [PredictionResponse, PredictionResponse];
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type Comparison = {
  "absolute_probability_difference": number;
  "ensemble_probability"?: null;
  "interpretation"?: "descriptive_only_not_model_confidence";
  "selected_winner"?: null;
};

export type ErrorBody = {
  "code": string;
  "details": Array<ErrorDetail>;
  "message": string;
};

export type ErrorDetail = {
  "field": string;
  "reason": string;
};

export type ErrorResponse = {
  "error": ErrorBody;
  "request_id": string;
};

export type ExplanationRequest = {
  "feature_context": FeatureContext;
  "feature_version": string;
  "features": PredictionFeatures;
  "model_key"?: string | null;
  "top_k"?: number;
};

export type ExplanationResponse = {
  "algorithm": "random_forest" | "xgboost";
  "api_version"?: "v1";
  "calibration_status"?: "not_operationally_validated";
  "explanation": ProbabilityExplanation | LogOddsExplanation;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "model_hash": string;
  "model_key": string;
  "model_version": string;
  "operational_use_allowed"?: false;
  "prediction_target"?: "NO_SHOW_CANDIDATE";
  "probability_no_show": number;
  "request_id": string;
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "target_class"?: 1;
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type Factor = {
  "absolute_contribution": number;
  "contribution": number;
  "direction": "increases_model_output" | "decreases_model_output" | "neutral";
  "feature_name": "age_at_booking" | "client_sex" | "booking_lead_days" | "appointment_month" | "appointment_weekday" | "appointment_hour" | "duration_minutes" | "is_fwa" | "clinic_id" | "scheduled_service_lines" | "distinct_body_areas" | "single_body_area_id" | "has_medical_evaluation" | "has_type4_service" | "previous_attended" | "previous_no_show" | "previous_no_show_rate" | "days_since_previous_attended" | "days_since_previous_no_show";
};

export type FeatureContext = {
  "builder_version"?: string | null;
  "snapshot_type": "booking_creation";
};

export type LiveResponse = {
  "api_version"?: "v1";
  "service"?: "depi-noshow-ml-api";
  "status"?: "alive";
};

export type LogOddsExplanation = {
  "additivity_error": number;
  "base_value": number;
  "explained_class"?: 1;
  "explanation_scope"?: "local_association_not_causality";
  "factors": Array<Factor>;
  "feature_perturbation"?: "tree_path_dependent";
  "method"?: "tree_shap";
  "model_output_in_explanation_space": number;
  "other_features_contribution": number;
  "output_link": "sigmoid";
  "output_space": "log_odds";
  "reconstructed_output_in_explanation_space": number;
  "top_k": number;
  "total_feature_count"?: 19;
};

export type ModelResponse = {
  "algorithm": "random_forest" | "xgboost";
  "calibration_status"?: "not_operationally_validated";
  "experiment"?: "full";
  "feature_count": number;
  "model_hash": string;
  "model_key": string;
  "model_version": string;
  "operational_use_allowed"?: false;
  "predictor_columns": Array<string>;
  "role": "provisional_demo_default" | "independent_comparator";
  "training_mode"?: "exploratory";
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type ModelsResponse = {
  "api_version"?: "v1";
  "default_model_key": string;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "models": [ModelResponse, ModelResponse];
};

export type PredictionFeatures = {
  "age_at_booking": number | null;
  "appointment_hour": number;
  "appointment_month": number;
  "appointment_weekday": number;
  "booking_lead_days": number;
  "client_sex": "M" | "F" | "X" | null;
  "clinic_id": number;
  "days_since_previous_attended": number | null;
  "days_since_previous_no_show": number | null;
  "distinct_body_areas": number;
  "duration_minutes": number | null;
  "has_medical_evaluation": boolean | null;
  "has_type4_service": boolean | null;
  "is_fwa": boolean | null;
  "previous_attended": number;
  "previous_no_show": number;
  "previous_no_show_rate": number | null;
  "scheduled_service_lines": number;
  "single_body_area_id": number | null;
};

export type PredictionRequest = {
  "feature_context": FeatureContext;
  "feature_version": string;
  "features": PredictionFeatures;
  "model_key"?: string | null;
};

export type PredictionResponse = {
  "algorithm": "random_forest" | "xgboost";
  "api_version"?: "v1";
  "calibration_status"?: "not_operationally_validated";
  "feature_version"?: "BOOKING_BASELINE_V1";
  "model_hash": string;
  "model_key": string;
  "model_version": string;
  "operational_use_allowed"?: false;
  "prediction_target"?: "NO_SHOW_CANDIDATE";
  "probability_no_show": number;
  "request_id": string;
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "target_class"?: 1;
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type ProbabilityExplanation = {
  "additivity_error": number;
  "base_value": number;
  "explained_class"?: 1;
  "explanation_scope"?: "local_association_not_causality";
  "factors": Array<Factor>;
  "feature_perturbation"?: "tree_path_dependent";
  "method"?: "tree_shap";
  "model_output_in_explanation_space": number;
  "other_features_contribution": number;
  "output_link": "identity";
  "output_space": "probability";
  "reconstructed_output_in_explanation_space": number;
  "top_k": number;
  "total_feature_count"?: 19;
};

export type ReadyResponse = {
  "models_loaded": [string, string];
  "operational_use_allowed"?: false;
  "serving_status"?: "EXPLORATORY_NOT_VALIDATED";
  "status"?: "ready";
};

export type SingleBatchItemResult = {
  "index": number;
  "prediction": PredictionResponse;
};

export type BatchRequest = BatchSingleRequest | BatchCompareRequest;
export type BatchResponse = BatchSingleResponse | BatchCompareResponse;
export type ShapExplanation = ProbabilityExplanation | LogOddsExplanation;
