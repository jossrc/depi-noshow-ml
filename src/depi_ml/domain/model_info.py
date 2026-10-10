"""Identidad inmutable del pipeline y política científica del MVP."""

from dataclasses import dataclass

FEATURE_VERSION = "BOOKING_BASELINE_V1"
VALIDATION_STATUS = "EXPLORATORY_NOT_VALIDATED"
CALIBRATION_STATUS = "not_operationally_validated"
MODEL_KEYS = ("random_forest_full_v1", "xgboost_full_v1")


@dataclass(frozen=True)
class ModelInfo:
    model_key: str
    algorithm: str
    model_sha256: str
    manifest_sha256: str
    training_source_sha256: str
    dataset_sha256: str
    features: tuple[str, ...]
    feature_version: str = FEATURE_VERSION
    experiment: str = "full"
    training_mode: str = "exploratory"
    validation_status: str = VALIDATION_STATUS
    calibration_status: str = CALIBRATION_STATUS
    operational_use_allowed: bool = False

    @property
    def model_version(self) -> str:
        return f"{self.model_key}:{self.model_sha256[:12]}"
