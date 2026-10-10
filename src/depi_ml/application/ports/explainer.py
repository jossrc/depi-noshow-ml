from typing import Mapping, Protocol

from depi_ml.application.dto.explanation_result import ExplanationResult


class Explainer(Protocol):
    def explain(self, features: Mapping[str, object], probability: float, top_k: int) -> ExplanationResult: ...
