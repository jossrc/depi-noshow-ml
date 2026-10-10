from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.errors import BasePredictionException


def error_response(status, code, message, request_id, details=None):
    return JSONResponse(status_code=status, content={
        "error": {"code": code, "message": message, "details": details or []}, "request_id": request_id})


def register_exception_handlers(app):
    @app.exception_handler(BasePredictionException)
    async def domain_error(request, error):
        return error_response(error.status_code, error.code, error.message, request.state.request_id,
                              getattr(error, "details", None))

    @app.exception_handler(RequestValidationError)
    async def invalid_features(request, error):
        # No raw input, exception text, arbitrary extra field names or Pydantic ctx.
        safe = {*PREDICTOR_COLUMNS, "features", "feature_context", "snapshot_type", "builder_version",
                "feature_version", "model_key", "body"}
        is_batch = request.url.path == "/internal/v1/predictions/batch"
        if is_batch:
            safe.update(("items", "mode"))
        elif request.url.path == "/internal/v1/predictions/explain":
            safe.add("top_k")

        def location(parts):
            field = ""
            for index, part in enumerate(parts):
                if part == "body" or is_batch and index == 1 and part in ("single", "compare"):
                    continue
                if is_batch and type(part) is int:
                    field += f"[{part}]"
                else:
                    field += ("." if field else "") + (str(part) if part in safe else "[extra]")
            return field
        details = [{"field": location(e["loc"]) or ("mode" if is_batch else ""),
                    "reason": e["type"]} for e in error.errors()[:20]]
        return error_response(422, "INVALID_FEATURES", "Los datos de predicción no cumplen el contrato v1.",
                              request.state.request_id, details)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        codes = {401: "UNAUTHORIZED_SERVICE", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        return error_response(error.status_code, codes.get(error.status_code, "INVALID_REQUEST"),
                              "La solicitud HTTP no puede procesarse.", request.state.request_id)
