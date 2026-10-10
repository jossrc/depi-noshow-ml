"""Errores públicos controlados; nunca contienen entradas ni rutas locales."""


class BasePredictionException(Exception):
    code = "INVALID_REQUEST"
    status_code = 400
    message = "La solicitud no cumple el contrato v1."


class ModelNotReady(BasePredictionException):
    code, status_code = "MODEL_NOT_READY", 503
    message = "La inferencia experimental no está disponible."


class UnknownModel(BasePredictionException):
    code, status_code = "UNKNOWN_MODEL", 404
    message = "El modelo solicitado no está registrado."


class FeatureVersionMismatch(BasePredictionException):
    code, status_code = "FEATURE_VERSION_MISMATCH", 409
    message = "La versión de características no es compatible."


class PredictionEngineException(BasePredictionException):
    code, status_code = "INFERENCE_FAILED", 500
    message = "No se pudo completar la inferencia."


class RateLimited(BasePredictionException):
    code, status_code = "RATE_LIMITED", 429
    message = "Se alcanzó el límite de inferencias simultáneas."


class ExplanationNotReady(BasePredictionException):
    code, status_code = "EXPLANATION_NOT_READY", 503
    message = "La explicación local del modelo no está disponible."


class ExplanationFailed(BasePredictionException):
    code, status_code = "EXPLANATION_FAILED", 500
    message = "No se pudo completar la explicación local."


class InvalidBatchSize(BasePredictionException):
    code, status_code = "INVALID_FEATURES", 422
    message = "La cantidad de elementos no cumple el límite configurado."
    details = ({"field": "items", "reason": "batch_size_out_of_range"},)
