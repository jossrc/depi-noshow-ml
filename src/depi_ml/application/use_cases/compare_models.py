from depi_ml.application.dto.prediction_input import PredictionInput
from depi_ml.application.ports.predictor import InferenceEngine
from depi_ml.application.use_cases.predict_no_show import validate_context


class CompareModels:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def execute(self, input: PredictionInput):
        validate_context(input)
        return self.engine.compare(input.features)
