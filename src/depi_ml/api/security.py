from secrets import compare_digest

from fastapi import Security
from fastapi.security import APIKeyHeader
from starlette.requests import Request
from starlette.exceptions import HTTPException

service_key_header = APIKeyHeader(name="X-Service-API-Key", auto_error=False)


def valid_service_key(settings, provided):
    return (settings.service_key is not None and isinstance(provided, str) and provided.isascii()
            and compare_digest(provided, settings.service_key.get_secret_value()))


def require_service_key(request: Request, key: str | None = Security(service_key_header)):
    if not valid_service_key(request.app.state.settings, key):
        raise HTTPException(401, "UNAUTHORIZED_SERVICE")
