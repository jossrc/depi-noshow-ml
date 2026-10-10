from typing import Protocol

from depi_ml.application.ports.predictor import Predictor
from depi_ml.domain.model_info import ModelInfo


class ModelRegistryPort(Protocol):
    def get(self, model_key: str) -> Predictor: ...

    def list_models(self) -> tuple[ModelInfo, ...]: ...

    @property
    def ready(self) -> bool: ...
