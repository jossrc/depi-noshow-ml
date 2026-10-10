from dataclasses import asdict

from depi_ml.api.mappers.prediction_mapper import prediction_response
from depi_ml.api.schemas.explanation import ExplanationResponse, ProbabilityExplanation, LogOddsExplanation, Factor


def explanation_response(result, request_id):
    explanation = result.explanation
    data = asdict(explanation)
    data["factors"] = tuple(Factor(**asdict(factor)) for factor in explanation.factors)
    schema = ProbabilityExplanation if explanation.output_space == "probability" else LogOddsExplanation
    return ExplanationResponse(**prediction_response(result.prediction, request_id).model_dump(),
                               explanation=schema(**data))
