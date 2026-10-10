from typing import Annotated
from fastapi import APIRouter, Depends, Request, Body

from depi_ml.api.dependencies import batch_use_case
from depi_ml.api.mappers.batch_mapper import batch_response
from depi_ml.api.schemas.batch import BatchSingleRequest, BatchCompareRequest, BatchResponse
from depi_ml.api.schemas.errors import ERROR_RESPONSES
from depi_ml.api.schemas.examples import batch_examples
from depi_ml.api.security import require_service_key
from depi_ml.application.dto.batch_input import BatchInput
from depi_ml.application.dto.prediction_input import PredictionInput

router = APIRouter(prefix="/internal/v1/predictions", tags=["predictions"],
                   dependencies=[Depends(require_service_key)])


@router.post("/batch", response_model=BatchResponse, responses=ERROR_RESPONSES, operation_id="PredictBatch",
             description="Validación atómica; ejecución ordenada bajo un cupo global. "
                         "Límite configurable ML_MAX_BATCH_ITEMS: 32 por defecto, máximo 64. Sin archivos ni IDs.")
def batch_handler(request: Request,
                  body: Annotated[BatchSingleRequest | BatchCompareRequest,
                                  Body(discriminator="mode", openapi_examples=batch_examples())],
                  use_case=Depends(batch_use_case)):
    items = tuple(PredictionInput(item.features.model_dump(), body.feature_version,
                                 body.feature_context.snapshot_type) for item in body.items)
    result = use_case.execute(BatchInput(body.mode, items, getattr(body, "model_key", None)))
    return batch_response(result, request.state.request_id)
