from dataclasses import dataclass
from typing import Literal

from depi_ml.application.dto.prediction_result import PredictionResult


@dataclass(frozen=True)
class ExplanationFactor:
    feature_name: str
    direction: Literal["increases_model_output", "decreases_model_output", "neutral"]
    contribution: float
    absolute_contribution: float


@dataclass(frozen=True)
class ExplanationResult:
    output_space: Literal["probability", "log_odds"]
    output_link: Literal["identity", "sigmoid"]
    base_value: float
    model_output_in_explanation_space: float
    reconstructed_output_in_explanation_space: float
    additivity_error: float
    top_k: int
    factors: tuple[ExplanationFactor, ...]
    other_features_contribution: float


@dataclass(frozen=True)
class ExplainedPrediction:
    prediction: PredictionResult
    explanation: ExplanationResult
