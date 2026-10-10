"""Guardia ASGI antes del parsing: auth, bytes reales, JSON y request-id acotado."""

import json
import logging
import re
from time import perf_counter
from uuid import uuid4

from depi_ml.api.exception_handlers import error_response
from depi_ml.api.security import valid_service_key
from depi_ml.infrastructure.telemetry.request_logging import request_id_context

logger = logging.getLogger(__name__)


class RequestGuard:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = [(k.lower(), v) for k, v in scope.get("headers", [])]
        ids = [v.decode("latin1") for k, v in headers if k == b"x-request-id"]
        request_id = ids[0] if len(ids) == 1 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", ids[0]) else uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        context_token = request_id_context.set(request_id)
        status, started, sent = 500, perf_counter(), False

        async def tracked_send(message):
            nonlocal status, sent
            if message["type"] == "http.response.start":
                sent, status = True, message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", request_id.encode())]}
            await send(message)

        async def reject(status_code, code, message):
            response = error_response(status_code, code, message, request_id)
            await response(scope, receive, tracked_send)

        try:
            internal = scope["path"] == "/internal" or scope["path"].startswith("/internal/")
            if internal:
                keys = [v.decode("latin1") for k, v in headers if k == b"x-service-api-key"]
                if len(keys) != 1 or not valid_service_key(self.settings, keys[0]):
                    return await reject(401, "UNAUTHORIZED_SERVICE", "Credencial de servicio ausente o inválida.")
            lengths = [v for k, v in headers if k == b"content-length"]
            if len(lengths) > 1 or (lengths and (not lengths[0].isdigit())):
                return await reject(400, "INVALID_REQUEST", "Content-Length inválido.")
            if lengths and int(lengths[0]) > self.settings.max_request_bytes:
                return await reject(413, "PAYLOAD_TOO_LARGE", "El cuerpo excede el límite permitido.")
            chunks, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > self.settings.max_request_bytes:
                    return await reject(413, "PAYLOAD_TOO_LARGE", "El cuerpo excede el límite permitido.")
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            if internal and scope["method"] == "POST":
                types = [v.decode("latin1").lower() for k, v in headers if k == b"content-type"]
                if len(types) != 1 or types[0].split(";")[0].strip() != "application/json":
                    return await reject(400, "INVALID_REQUEST", "Se requiere application/json UTF-8.")
                try:
                    json.loads(body.decode("utf-8"), parse_constant=reject_nonfinite,
                               object_pairs_hook=unique_json_object)
                except NonFiniteJson:
                    return await reject(422, "INVALID_FEATURES", "Los números deben ser finitos.")
                except (ValueError, UnicodeError, RecursionError):
                    return await reject(400, "INVALID_REQUEST", "JSON inválido.")
            delivered = False

            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, tracked_send)
        except Exception:
            if sent:
                raise
            logger.error("event=request_failed request_id=%s error_code=INFERENCE_FAILED", request_id)
            await reject(500, "INFERENCE_FAILED", "No se pudo completar la solicitud.")
        finally:
            route = scope.get("route")
            path = getattr(route, "path", "unmatched")
            logger.info("event=http_request request_id=%s route=%s status=%d elapsed_ms=%.3f",
                        request_id, path, status, (perf_counter() - started) * 1000)
            request_id_context.reset(context_token)


class NonFiniteJson(ValueError):
    pass


def reject_nonfinite(value):
    raise NonFiniteJson


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result
