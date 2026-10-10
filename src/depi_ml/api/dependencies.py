from starlette.requests import Request

from depi_ml.domain.errors import ModelNotReady


def require_registry(request: Request):
    registry = request.app.state.registry
    if registry is None or not registry.ready:
        raise ModelNotReady
    return registry


def prediction_use_case(request: Request):
    require_registry(request)
    return request.app.state.predict_no_show


def comparison_use_case(request: Request):
    require_registry(request)
    return request.app.state.compare_models
