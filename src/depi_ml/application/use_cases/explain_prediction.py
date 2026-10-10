from depi_ml.application.dto.explanation_input import ExplanationInput
from depi_ml.application.ports.predictor import InferenceEngine
from depi_ml.application.use_cases.predict_no_show import validate_context
from depi_ml.domain.errors import BasePredictionException
from depi_ml.domain.model_info import MODEL_KEYS


class ExplainPrediction:
    def __init__(self, engine: InferenceEngine, default_model_key=MODEL_KEYS[0]):
        self.engine, self.default_model_key = engine, default_model_key

    def execute(self, input: ExplanationInput):
        validate_context(input.prediction)
        if type(input.top_k) is not int or not 1 <= input.top_k <= 10:
            raise BasePredictionException
        return self.engine.explain(input.prediction.model_key or self.default_model_key,
                                   input.prediction.features, input.top_k)
