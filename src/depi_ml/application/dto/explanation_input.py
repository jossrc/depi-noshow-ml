from dataclasses import dataclass

from depi_ml.application.dto.prediction_input import PredictionInput


@dataclass(frozen=True)
class ExplanationInput:
    prediction: PredictionInput
    top_k: int = 5
