"""Superficie v2 aditiva. Reutiliza requests/19 features y mantiene schemas v1."""

from typing import Annotated, Literal

from pydantic import Field

from depi_ml.api.schemas.prediction import PredictionResponse, CompareResponse, StrictObject
from depi_ml.domain.research_risk_policy import Sha256, Finite, Level
Threshold = Annotated[float, Field(gt=0,lt=1,allow_inf_nan=False)]


class ExperimentalRisk(StrictObject):
    level: Level
    label: Literal["Bajo (exploratorio)", "Medio (exploratorio)", "Alto (exploratorio)"]
    status: Literal["RESEARCH_PROVISIONAL"] = "RESEARCH_PROVISIONAL"
    policy_id: str
    policy_version: Literal["research_risk_v1_1"]
    policy_hash: Sha256
    score_space: Literal["raw_probability_no_show"] = "raw_probability_no_show"
    low_upper_threshold: Threshold
    high_lower_threshold: Threshold
    boundary_rule: Literal["LOW:p<t_low;MEDIUM:t_low<=p<t_high;HIGH:p>=t_high"]
    meaning: Literal["relative_score_band_not_validated_absolute_risk"] = "relative_score_band_not_validated_absolute_risk"
    operational_use_allowed: Literal[False] = False
    approved_by: None = None
    warnings: tuple[str, ...]


class ResearchPredictionResponse(PredictionResponse):
    api_version: Literal["v2"] = "v2"
    model_hash_sha256: Sha256
    probability_no_show_percent: Annotated[float, Field(ge=0,le=100,allow_inf_nan=False)]
    experimental_risk: ExperimentalRisk


class ResearchCompareResponse(CompareResponse):
    api_version: Literal["v2"] = "v2"
    results: tuple[ResearchPredictionResponse, ResearchPredictionResponse]


class ResearchPolicyView(StrictObject):
    model_key: str
    model_version: str
    model_hash_sha256: Sha256
    feature_version: Literal["BOOKING_BASELINE_V1"] = "BOOKING_BASELINE_V1"
    policy_id: str
    policy_version: Literal["research_risk_v1_1"]
    policy_hash: Sha256
    low_upper_threshold: Threshold
    high_lower_threshold: Threshold
    boundary_rule: Literal["LOW:p<t_low;MEDIUM:t_low<=p<t_high;HIGH:p>=t_high"]
    method: Literal["score_quantiles_tertiles_exploratory"]
    meaning: Literal["relative_score_band_not_validated_absolute_risk"]
    development_start: str
    development_end_exclusive: str
    label_cutoff_exclusive: str
    development_rows: int
    calibration_ece_uniform_10: Finite
    status: Literal["RESEARCH_PROVISIONAL"] = "RESEARCH_PROVISIONAL"
    operational_use_allowed: Literal[False] = False
    approved_by: None = None
    warnings: tuple[str, ...]


class ResearchPolicyCatalog(StrictObject):
    api_version: Literal["v2"] = "v2"
    status: Literal["RESEARCH_PROVISIONAL", "UNAVAILABLE"]
    available: bool
    reason: Literal["disabled", "models_unavailable", "missing_policy", "hash_mismatch",
                    "invalid_policy", "model_identity_mismatch", "insufficient_support"] | None
    policies: tuple[ResearchPolicyView, ...]
    validation_status: Literal["EXPLORATORY_NOT_VALIDATED"] = "EXPLORATORY_NOT_VALIDATED"
    operational_use_allowed: Literal[False] = False
    request_id: str
