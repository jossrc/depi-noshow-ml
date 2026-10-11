// Generated v2 research contract; operational_use_allowed=false.
export type CompareRequest = {
  "feature_context": FeatureContext;
  "feature_version": string;
  "features": PredictionFeatures;
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

export type ExperimentalRisk = {
  "approved_by"?: null;
  "boundary_rule": "LOW:p<t_low;MEDIUM:t_low<=p<t_high;HIGH:p>=t_high";
  "high_lower_threshold": number;
  "label": "Bajo (exploratorio)" | "Medio (exploratorio)" | "Alto (exploratorio)";
  "level": "LOW" | "MEDIUM" | "HIGH";
  "low_upper_threshold": number;
  "meaning"?: "relative_score_band_not_validated_absolute_risk";
  "operational_use_allowed"?: false;
  "policy_hash": string;
  "policy_id": string;
  "policy_version": "research_risk_v1_1";
  "score_space"?: "raw_probability_no_show";
  "status"?: "RESEARCH_PROVISIONAL";
  "warnings": Array<string>;
};

export type FeatureContext = {
  "builder_version"?: string | null;
  "snapshot_type": "booking_creation";
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

export type ResearchCompareResponse = {
  "api_version"?: "v2";
  "calibration_status"?: "not_operationally_validated";
  "comparison": Comparison;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "operational_use_allowed"?: false;
  "request_id": string;
  "results": [ResearchPredictionResponse, ResearchPredictionResponse];
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type ResearchPolicyCatalog = {
  "api_version"?: "v2";
  "available": boolean;
  "operational_use_allowed"?: false;
  "policies": Array<ResearchPolicyView>;
  "reason": "disabled" | "models_unavailable" | "missing_policy" | "hash_mismatch" | "invalid_policy" | "model_identity_mismatch" | "insufficient_support" | null;
  "request_id": string;
  "status": "RESEARCH_PROVISIONAL" | "UNAVAILABLE";
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};

export type ResearchPolicyView = {
  "approved_by"?: null;
  "boundary_rule": "LOW:p<t_low;MEDIUM:t_low<=p<t_high;HIGH:p>=t_high";
  "calibration_ece_uniform_10": number;
  "development_end_exclusive": string;
  "development_rows": number;
  "development_start": string;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "high_lower_threshold": number;
  "label_cutoff_exclusive": string;
  "low_upper_threshold": number;
  "meaning": "relative_score_band_not_validated_absolute_risk";
  "method": "score_quantiles_tertiles_exploratory";
  "model_hash_sha256": string;
  "model_key": string;
  "model_version": string;
  "operational_use_allowed"?: false;
  "policy_hash": string;
  "policy_id": string;
  "policy_version": "research_risk_v1_1";
  "status"?: "RESEARCH_PROVISIONAL";
  "warnings": Array<string>;
};

export type ResearchPredictionResponse = {
  "algorithm": "random_forest" | "xgboost";
  "api_version"?: "v2";
  "calibration_status"?: "not_operationally_validated";
  "experimental_risk": ExperimentalRisk;
  "feature_version"?: "BOOKING_BASELINE_V1";
  "model_hash": string;
  "model_hash_sha256": string;
  "model_key": string;
  "model_version": string;
  "operational_use_allowed"?: false;
  "prediction_target"?: "NO_SHOW_CANDIDATE";
  "probability_no_show": number;
  "probability_no_show_percent": number;
  "request_id": string;
  "risk_level"?: null;
  "risk_policy_status"?: "pending_validation";
  "target_class"?: 1;
  "validation_status"?: "EXPLORATORY_NOT_VALIDATED";
};
