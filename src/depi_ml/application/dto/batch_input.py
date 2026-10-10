from dataclasses import dataclass
from typing import Literal

from depi_ml.application.dto.prediction_input import PredictionInput


@dataclass(frozen=True)
class BatchInput:
    mode: Literal["single", "compare"]
    items: tuple[PredictionInput, ...]
    model_key: str | None = None
