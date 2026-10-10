import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError


def test_import_does_not_import_database_joblib_or_pandas():
    code = "import sys; import depi_ml.api.main; assert not any(m in sys.modules for m in ['psycopg', 'joblib', 'pandas', 'depi_ml.config', 'depi_ml.db.connection'])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_liveness_and_ready(client):
    assert client.get("/health/live").json() == {"status": "alive", "service": "depi-noshow-ml-api", "api_version": "v1"}
    assert client.get("/health/ready").json()["models_loaded"] == list(MODEL_KEYS)
    assert client.get("/health/ready").json()["operational_use_allowed"] is False


@pytest.mark.parametrize("settings", [ApiSettings(), ApiSettings(env="production", allow_experimental_inference=True,
                       service_key="synthetic-unit-service-key-not-a-secret-2026"),
                       ApiSettings(allow_experimental_inference=True)])
def test_disabled_demo_never_loads_models(settings):
    def forbidden(_):
        pytest.fail("No artifact load allowed")
    with TestClient(create_app(settings, registry_loader=forbidden)) as c:
        assert c.get("/health/live").status_code == 200
        assert c.get("/health/ready").status_code == 503
        assert c.get("/docs").status_code == 404


@pytest.mark.parametrize("failure", [ArtifactVerificationError("model_hash_mismatch"), RuntimeError("/secret/local/path")])
def test_startup_failure_sanitized(settings, failure, headers, payload, caplog):
    def load(_):
        raise failure
    with TestClient(create_app(settings, registry_loader=load)) as c:
        assert c.get("/health/live").status_code == 200
        r = c.post("/internal/v1/predictions/compare", headers=headers, json=payload)
        assert r.status_code == 503
        assert "/secret" not in r.text + caplog.text


def test_missing_xgb_readiness_fails(settings, registry):
    registry.predictors.pop(MODEL_KEYS[1])
    with TestClient(create_app(settings, registry_loader=lambda _: registry)) as c:
        assert c.get("/health/ready").status_code == 503


def test_loader_once_and_shutdown_restart(settings, registry, payload, headers):
    calls = []
    def load(_):
        calls.append(1)
        return registry
    app = create_app(settings, registry_loader=load)
    assert calls == [] and app.state.registry is None
    for count in (1, 2):
        with TestClient(app) as c:
            for _ in range(2):
                assert c.post("/internal/v1/predictions/compare", json=payload, headers=headers).status_code == 200
            assert len(calls) == count
        assert app.state.registry is None


def test_no_db_or_file_load_on_requests(client, headers, payload, monkeypatch):
    import builtins
    import psycopg
    def forbidden(*args, **kwargs):
        pytest.fail("No DB/file access on request")
    monkeypatch.setattr(psycopg, "connect", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    for path in ("/internal/v1/predictions", "/internal/v1/predictions/compare"):
        assert client.post(path, headers=headers, json=payload).status_code == 200


def test_environment_does_not_read_dotenv_or_require_password(monkeypatch):
    import dotenv
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: pytest.fail("No dotenv loading"))
    assert isinstance(ApiSettings.from_env(), ApiSettings)


def test_invalid_env_secret_not_echoed(monkeypatch):
    monkeypatch.setenv("ML_API_SERVICE_KEY", "sensitive")
    with pytest.raises(ValueError) as error:
        ApiSettings.from_env()
    assert "sensitive" not in str(error.value)
