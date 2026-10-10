from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from depi_ml.api.schemas.features import PredictionFeatures
from depi_ml.domain.model_info import FEATURE_VERSION, CALIBRATION_STATUS, VALIDATION_STATUS


class StrictObject(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class FeatureContext(StrictObject):
    snapshot_type: Literal["booking_creation"]
    builder_version: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")] | None = None


class CompareRequest(StrictObject):
    feature_version: Annotated[str, Field(min_length=1, max_length=64)]
    feature_context: FeatureContext
    features: PredictionFeatures


class PredictionRequest(CompareRequest):
    model_key: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")] | None = None


class PredictionResponse(StrictObject):
    api_version: Literal["v1"] = "v1"
    model_key: str
    algorithm: Literal["random_forest", "xgboost"]
    model_version: str
    model_hash: str
    feature_version: Literal["BOOKING_BASELINE_V1"] = FEATURE_VERSION
    probability_no_show: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    prediction_target: Literal["NO_SHOW_CANDIDATE"] = "NO_SHOW_CANDIDATE"
    target_class: Literal[1] = 1
    risk_level: None = None
    risk_policy_status: Literal["pending_validation"] = "pending_validation"
    validation_status: Literal["EXPLORATORY_NOT_VALIDATED"] = VALIDATION_STATUS
    calibration_status: Literal["not_operationally_validated"] = CALIBRATION_STATUS
    operational_use_allowed: Literal[False] = False
    request_id: str


class Comparison(StrictObject):
    absolute_probability_difference: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    interpretation: Literal["descriptive_only_not_model_confidence"] = "descriptive_only_not_model_confidence"
    ensemble_probability: None = None
    selected_winner: None = None


class CompareResponse(StrictObject):
    api_version: Literal["v1"] = "v1"
    feature_version: Literal["BOOKING_BASELINE_V1"] = FEATURE_VERSION
    results: tuple[PredictionResponse, PredictionResponse]
    comparison: Comparison
    risk_level: None = None
    risk_policy_status: Literal["pending_validation"] = "pending_validation"
    validation_status: Literal["EXPLORATORY_NOT_VALIDATED"] = VALIDATION_STATUS
    calibration_status: Literal["not_operationally_validated"] = CALIBRATION_STATUS
    operational_use_allowed: Literal[False] = False
    request_id: str
