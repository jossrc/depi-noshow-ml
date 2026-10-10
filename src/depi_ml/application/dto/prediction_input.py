from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class PredictionInput:
    features: Mapping[str, object]
    feature_version: str
    snapshot_type: str
    model_key: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
