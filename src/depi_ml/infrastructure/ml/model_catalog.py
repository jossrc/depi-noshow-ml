from types import MappingProxyType

from depi_ml.domain.model_info import MODEL_KEYS

MODEL_CATALOG = MappingProxyType(dict(zip(MODEL_KEYS, ("random_forest", "xgboost"), strict=True)))
