from typing import Literal

from depi_ml.api.schemas.prediction import StrictObject


class ModelResponse(StrictObject):
    model_key: str
    algorithm: Literal["random_forest", "xgboost"]
    model_version: str
    model_hash: str
    experiment: Literal["full"] = "full"
    training_mode: Literal["exploratory"] = "exploratory"
    validation_status: Literal["EXPLORATORY_NOT_VALIDATED"] = "EXPLORATORY_NOT_VALIDATED"
    calibration_status: Literal["not_operationally_validated"] = "not_operationally_validated"
    operational_use_allowed: Literal[False] = False
    feature_count: int
    predictor_columns: tuple[str, ...]
    role: Literal["provisional_demo_default", "independent_comparator"]


class ModelsResponse(StrictObject):
    api_version: Literal["v1"] = "v1"
    feature_version: Literal["BOOKING_BASELINE_V1"] = "BOOKING_BASELINE_V1"
    default_model_key: str
    models: tuple[ModelResponse, ModelResponse]


class LiveResponse(StrictObject):
    status: Literal["alive"] = "alive"
    service: Literal["depi-noshow-ml-api"] = "depi-noshow-ml-api"
    api_version: Literal["v1"] = "v1"


class ReadyResponse(StrictObject):
    status: Literal["ready"] = "ready"
    models_loaded: tuple[str, str]
    serving_status: Literal["EXPLORATORY_NOT_VALIDATED"] = "EXPLORATORY_NOT_VALIDATED"
    operational_use_allowed: Literal[False] = False
