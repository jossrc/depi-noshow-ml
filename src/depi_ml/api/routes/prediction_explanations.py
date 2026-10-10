from fastapi import APIRouter, Depends, Request, Body

from depi_ml.api.dependencies import explanation_use_case
from depi_ml.api.mappers.explanation_mapper import explanation_response
from depi_ml.api.routes.predictions import to_input
from depi_ml.api.schemas.explanation import ExplanationRequest, ExplanationResponse
from depi_ml.api.schemas.errors import ERROR_RESPONSES
from depi_ml.api.security import require_service_key
from depi_ml.api.schemas.examples import explanation_examples
from depi_ml.application.dto.explanation_input import ExplanationInput

router = APIRouter(prefix="/internal/v1/predictions", tags=["predictions"],
                   dependencies=[Depends(require_service_key)])


@router.post("/explain", response_model=ExplanationResponse, responses=ERROR_RESPONSES,
             operation_id="ExplainPrediction",
             description="Tree SHAP local opcional. RF: probability/identity; XGB: log_odds/sigmoid. "
                         "Una explicación simultánea; no causalidad ni uso operativo aprobado.")
def explain_handler(request: Request, body: ExplanationRequest = Body(openapi_examples=explanation_examples()),
                    use_case=Depends(explanation_use_case)):
    result = use_case.execute(ExplanationInput(to_input(body, body.model_key), body.top_k))
    return explanation_response(result, request.state.request_id)
