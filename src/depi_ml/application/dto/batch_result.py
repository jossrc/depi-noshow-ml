from dataclasses import dataclass
from typing import Literal

from depi_ml.application.dto.prediction_result import PredictionResult


@dataclass(frozen=True)
class BatchResult:
    mode: Literal["single", "compare"]
    predictions: tuple[PredictionResult | tuple[PredictionResult, ...], ...]
