from depi_ml.application.dto.prediction_input import PredictionInput
from depi_ml.application.ports.predictor import InferenceEngine
from depi_ml.domain.errors import FeatureVersionMismatch, BasePredictionException
from depi_ml.domain.model_info import FEATURE_VERSION, MODEL_KEYS
from depi_ml.datasets.schema import PREDICTOR_COLUMNS


def validate_context(input: PredictionInput):
    if input.feature_version != FEATURE_VERSION:
        raise FeatureVersionMismatch
    if input.snapshot_type != "booking_creation" or set(input.features) != set(PREDICTOR_COLUMNS):
        raise BasePredictionException


class PredictNoShow:
    def __init__(self, engine: InferenceEngine, default_model_key: str = MODEL_KEYS[0]):
        self.engine, self.default_model_key = engine, default_model_key

    def execute(self, input: PredictionInput):
        validate_context(input)
        return self.engine.predict(input.model_key or self.default_model_key, input.features)
