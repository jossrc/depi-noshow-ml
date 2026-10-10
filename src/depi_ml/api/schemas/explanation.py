from typing import Annotated, Literal
from pydantic import Field

from depi_ml.api.schemas.prediction import PredictionRequest, PredictionResponse, StrictObject
from depi_ml.datasets.schema import PREDICTOR_COLUMNS

Finite = Annotated[float, Field(allow_inf_nan=False)]


class ExplanationRequest(PredictionRequest):
    top_k: Annotated[int, Field(ge=1, le=10)] = 5


class Factor(StrictObject):
    feature_name: Literal[tuple(PREDICTOR_COLUMNS)]
    direction: Literal["increases_model_output", "decreases_model_output", "neutral"]
    contribution: Finite
    absolute_contribution: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class LocalExplanation(StrictObject):
    method: Literal["tree_shap"] = "tree_shap"
    feature_perturbation: Literal["tree_path_dependent"] = "tree_path_dependent"
    base_value: Finite
    model_output_in_explanation_space: Finite
    reconstructed_output_in_explanation_space: Finite
    additivity_error: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    explained_class: Literal[1] = 1
    total_feature_count: Literal[19] = 19
    top_k: Annotated[int, Field(ge=1, le=10)]
    factors: Annotated[tuple[Factor, ...], Field(min_length=1, max_length=10)]
    other_features_contribution: Finite
    explanation_scope: Literal["local_association_not_causality"] = "local_association_not_causality"


class ProbabilityExplanation(LocalExplanation):
    output_space: Literal["probability"]
    output_link: Literal["identity"]


class LogOddsExplanation(LocalExplanation):
    output_space: Literal["log_odds"]
    output_link: Literal["sigmoid"]


class ExplanationResponse(PredictionResponse):
    explanation: Annotated[ProbabilityExplanation | LogOddsExplanation, Field(discriminator="output_space")]
