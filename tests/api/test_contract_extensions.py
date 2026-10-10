import json
import subprocess
import sys
from pathlib import Path
from pydantic import TypeAdapter
from depi_ml.api.main import create_app
from depi_ml.api.schemas.batch import BatchRequest
from depi_ml.api.schemas.explanation import ExplanationRequest
from depi_ml.infrastructure.config.api_settings import ApiSettings
from scripts.export_ml_contract import render_types

ROOT = Path(__file__).resolve().parents[2]

def test_export_snapshot_and_generated_types_are_current():
    schema = create_app(ApiSettings(env="development", expose_docs=True)).openapi()
    snapshot = json.loads((ROOT / "docs/contracts/ml-internal-v1.openapi.json").read_text())
    assert schema == snapshot
    assert (ROOT / "docs/contracts/ml-internal-v1.types.ts").read_text() == render_types(snapshot)
    assert schema["components"]["schemas"]["PredictionFeatures"]["additionalProperties"] is False
    assert len(schema["components"]["schemas"]["PredictionFeatures"]["required"]) == 19

def test_all_original_routes_and_schemas_are_unchanged():
    old = json.loads((ROOT / "tests/api/fixtures/prompt7_openapi.json").read_text())
    new = create_app(ApiSettings(expose_docs=True)).openapi()
    for path, definition in old["paths"].items():
        assert new["paths"][path] == definition
    for name, definition in old["components"]["schemas"].items():
        assert new["components"]["schemas"][name] == definition
    assert set(new["paths"]) - set(old["paths"]) == {
        "/internal/v1/predictions/explain", "/internal/v1/predictions/batch"}

def test_examples_validate_against_actual_models_and_swagger():
    schema = create_app(ApiSettings(expose_docs=True)).openapi()
    for operation, names, adapter in [
        ("explain", ("rf", "xgb"), TypeAdapter(ExplanationRequest)),
        ("batch", ("single", "compare"), TypeAdapter(BatchRequest))]:
        examples = schema["paths"]["/internal/v1/predictions/" + operation]["post"]["requestBody"]["content"]["application/json"]["examples"]
        for name in names:
            body = json.loads((ROOT / ("examples/prediction_" + operation + "_" + name + ".json")).read_text())
            adapter.validate_python(body)
            assert body == examples[name]["value"]
            assert "X-Service-API-Key" not in json.dumps(body)

def test_cli_and_api_factory_import_with_shap_blocked():
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *a, **k):
    if name == 'shap' or name.startswith('shap.'):
        raise ImportError('SHAP deliberately unavailable')
    return original(name, *a, **k)
builtins.__import__ = guarded
import depi_ml.cli
from depi_ml.api.main import create_app
from depi_ml.infrastructure.config.api_settings import ApiSettings
create_app(ApiSettings()).openapi()
assert 'shap' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_old_validation_extra_names_remain_sanitized(client, payload, headers):
    for name in ("top_k", "mode", "items"):
        r = client.post("/internal/v1/predictions", json={**payload, name: "private"}, headers=headers)
        assert r.status_code == 422
        assert r.json()["error"]["details"] == [{"field": "[extra]", "reason": "extra_forbidden"}]
