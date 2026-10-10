import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.api.middleware import RequestGuard


@pytest.mark.parametrize("path", ["/internal/v1/models", "/internal/v1/predictions", "/internal/v1/predictions/compare"])
@pytest.mark.parametrize("key", [None, "incorrect", "ñ"])
def test_service_auth(client, path, key):
    headers = [] if key is None else [(b"X-Service-API-Key", key.encode("utf-8"))]
    r = client.get(path, headers=headers) if path.endswith("models") else client.post(path, headers=headers, content="garbage")
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED_SERVICE"


def test_duplicate_auth_headers_rejected(client, headers):
    key = headers["X-Service-API-Key"]
    assert client.get("/internal/v1/models", headers=[("X-Service-API-Key", key)] * 2).status_code == 401


def test_authenticated_model_metadata(client, headers):
    r = client.get("/internal/v1/models", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert len(data["models"]) == 2
    assert all(m["feature_count"] == 19 and m["operational_use_allowed"] is False for m in data["models"])
    assert "model_path" not in r.text and "dataset_sha256" not in r.text


@pytest.mark.parametrize("request_id", ["x" * 65, "unsafe id", "foo/bar", ""])
def test_unsafe_request_ids_replaced(client, headers, request_id):
    r = client.get("/health/live", headers={"X-Request-ID": request_id})
    assert len(r.headers["X-Request-ID"]) == 32
    assert r.headers["X-Request-ID"] != request_id


def test_safe_logs_without_inputs_key_or_paths(client, headers, payload, caplog):
    caplog.set_level(logging.INFO, logger="depi_ml")
    payload["features"]["extra-sensitive-field"] = "sensitive-value"
    r = client.post("/internal/v1/predictions", headers=headers, json=payload)
    assert r.status_code == 422
    client.get("/client-sensitive-path")
    assert not any(s in caplog.text + r.text for s in [headers["X-Service-API-Key"], "sensitive-value", "extra-sensitive-field", "client-sensitive-path"])
    assert "elapsed_ms=" in caplog.text and "request_id=" in caplog.text


@pytest.mark.parametrize("content,ctype,expected", [("{", "application/json", 400), ("{}", "text/plain", 400),
                           ("null", "application/json", 422), (b"\xff", "application/json", 400)])
def test_invalid_body(client, headers, content, ctype, expected):
    r = client.post("/internal/v1/predictions", headers={**headers, "Content-Type": ctype}, content=content)
    assert r.status_code == expected


def test_body_size_before_parsing(settings, registry, headers):
    app = create_app(settings.model_copy(update={"max_request_bytes": 1024}), registry_loader=lambda _: registry)
    with TestClient(app) as c:
        r = c.post("/internal/v1/predictions", headers={**headers, "Content-Type": "application/json"}, content="x" * 1025)
        assert r.status_code == 413


def test_chunked_size_not_dependent_on_content_length(settings):
    messages = iter([{"type": "http.request", "body": b"x" * 700, "more_body": True},
                     {"type": "http.request", "body": b"x" * 700, "more_body": False}])
    output = []
    async def receive():
        return next(messages)
    async def send(message):
        output.append(message)
    async def forbidden(*args):
        pytest.fail("Must reject before reaching parser")
    guard = RequestGuard(forbidden, settings.model_copy(update={"max_request_bytes": 1024}))
    asyncio.run(guard({"type": "http", "path": "/health/live", "method": "POST", "headers": []}, receive, send))
    assert output[0]["status"] == 413


def test_uniform_500_for_unexpected_response_error(client, registry, payload, headers):
    from dataclasses import replace
    predictor = registry.get("random_forest_full_v1")
    predictor.info = replace(predictor.info, algorithm="invalid")
    r = client.post("/internal/v1/predictions", headers=headers, json=payload)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INFERENCE_FAILED"
    assert "Traceback" not in r.text


def test_openapi_and_no_unrequested_endpoints(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health/live", "/health/ready", "/internal/v1/models",
                                     "/internal/v1/predictions", "/internal/v1/predictions/compare",
                                     "/internal/v1/predictions/explain", "/internal/v1/predictions/batch"}
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-Service-API-Key"
    features = schema["components"]["schemas"]["PredictionFeatures"]
    assert features["additionalProperties"] is False
    assert len(features["required"]) == 19
    assert "client_id" not in json.dumps(schema)
    assert schema["paths"]["/internal/v1/predictions"]["post"]["responses"]["422"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


def test_duplicate_json_keys_rejected(client, headers):
    r = client.post("/internal/v1/predictions", headers={**headers, "Content-Type": "application/json"},
                    content='{"features": {}, "features": {}}')
    assert r.status_code == 400


def test_capacity_error_uniform_http(client, registry, headers, payload):
    from depi_ml.domain.errors import RateLimited
    registry.get("random_forest_full_v1").probability = RateLimited()
    r = client.post("/internal/v1/predictions", headers=headers, json=payload)
    assert r.status_code == 429 and r.json()["error"]["code"] == "RATE_LIMITED"
