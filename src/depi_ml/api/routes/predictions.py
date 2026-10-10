from fastapi import APIRouter, Depends, Request

from depi_ml.api.dependencies import prediction_use_case, comparison_use_case
from depi_ml.api.mappers.prediction_mapper import prediction_response, compare_response
from depi_ml.api.schemas.errors import ERROR_RESPONSES
from depi_ml.api.schemas.prediction import PredictionRequest, PredictionResponse, CompareRequest, CompareResponse
from depi_ml.api.security import require_service_key
from depi_ml.application.dto.prediction_input import PredictionInput
from depi_ml.datasets.schema import PREDICTOR_COLUMNS

PredictNoShowRoute = APIRouter(prefix="/internal/v1/predictions", tags=["predictions"],
                              dependencies=[Depends(require_service_key)])
router = PredictNoShowRoute


def to_input(body, model_key=None):
    return PredictionInput({name: getattr(body.features, name) for name in PREDICTOR_COLUMNS},
                           body.feature_version, body.feature_context.snapshot_type, model_key)


@router.post("", response_model=PredictionResponse, responses=ERROR_RESPONSES, operation_id="PredictNoShow")
def PredictNoShowHandler(body: PredictionRequest, request: Request, use_case=Depends(prediction_use_case)):
    return prediction_response(use_case.execute(to_input(body, body.model_key)), request.state.request_id)


@router.post("/compare", response_model=CompareResponse, responses=ERROR_RESPONSES, operation_id="CompareModels")
def compare_handler(body: CompareRequest, request: Request, use_case=Depends(comparison_use_case)):
    return compare_response(use_case.execute(to_input(body)), request.state.request_id)
