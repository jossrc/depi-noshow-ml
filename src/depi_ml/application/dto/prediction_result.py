from dataclasses import dataclass

from depi_ml.domain.model_info import ModelInfo


@dataclass(frozen=True)
class PredictionResult:
    info: ModelInfo
    probability_no_show: float
    risk_level: None = None
