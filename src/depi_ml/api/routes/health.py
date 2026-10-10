from fastapi import APIRouter, Depends

from depi_ml.api.dependencies import require_registry
from depi_ml.api.schemas.errors import ERROR_RESPONSES
from depi_ml.api.schemas.models import LiveResponse, ReadyResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LiveResponse)
def live():
    return LiveResponse()


@router.get("/ready", response_model=ReadyResponse, responses={503: ERROR_RESPONSES[503]})
def ready(registry=Depends(require_registry)):
    return ReadyResponse(models_loaded=tuple(info.model_key for info in registry.list_models()))
