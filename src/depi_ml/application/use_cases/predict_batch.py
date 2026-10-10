from depi_ml.application.dto.batch_input import BatchInput
from depi_ml.application.ports.predictor import InferenceEngine
from depi_ml.application.use_cases.predict_no_show import validate_context
from depi_ml.domain.errors import BasePredictionException, InvalidBatchSize
from depi_ml.domain.model_info import MODEL_KEYS


class PredictBatch:
    def __init__(self, engine: InferenceEngine, max_items=32, default_model_key=MODEL_KEYS[0]):
        self.engine, self.max_items, self.default_model_key = engine, max_items, default_model_key

    def execute(self, input: BatchInput):
        if not 1 <= len(input.items) <= self.max_items:
            raise InvalidBatchSize
        if input.mode not in ("single", "compare") or input.mode == "compare" and input.model_key is not None:
            raise BasePredictionException
        for item in input.items:
            validate_context(item)
        return self.engine.batch(input.mode, tuple(item.features for item in input.items),
                                 input.model_key or self.default_model_key)
