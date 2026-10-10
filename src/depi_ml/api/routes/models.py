from fastapi import APIRouter, Depends, Request

from depi_ml.api.dependencies import require_registry
from depi_ml.api.schemas.errors import ERROR_RESPONSES
from depi_ml.api.schemas.models import ModelsResponse, ModelResponse
from depi_ml.api.security import require_service_key

router = APIRouter(prefix="/internal/v1", tags=["models"], dependencies=[Depends(require_service_key)])


@router.get("/models", response_model=ModelsResponse, responses=ERROR_RESPONSES)
def models(request: Request, registry=Depends(require_registry)):
    default = request.app.state.settings.default_model_key
    return ModelsResponse(default_model_key=default, models=tuple(
        ModelResponse(model_key=info.model_key, algorithm=info.algorithm, model_version=info.model_version,
                      model_hash=info.model_sha256[:12], predictor_columns=info.features, feature_count=len(info.features),
                      role="provisional_demo_default" if info.model_key == default else "independent_comparator")
        for info in registry.list_models()))
