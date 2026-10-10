from typing import Mapping, Protocol

from depi_ml.application.dto.prediction_result import PredictionResult
from depi_ml.domain.model_info import ModelInfo


class Predictor(Protocol):
    info: ModelInfo

    def predict_probability(self, features: Mapping[str, object]) -> float: ...


class InferenceEngine(Protocol):
    def predict(self, model_key: str, features: Mapping[str, object]) -> PredictionResult: ...

    def compare(self, features: Mapping[str, object]) -> tuple[PredictionResult, ...]: ...
