"""Opt-in real: políticas publicadas, originales y fixtures artificiales, sin CSV/SQL."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.infrastructure.config.api_settings import ApiSettings

pytestmark=pytest.mark.skipif(os.environ.get("ML_RUN_REAL_INTEGRATION")!="1",reason="Original models opt-in")
ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def real_research_client():
    reference=Path(os.environ.get("ML_MODEL_REFERENCE_DIR","experiments/phase2-exploratory-prompt5-20261008"))
    if not (reference/"experiment.json").is_file():
        pytest.skip("Original private artifacts absent")
    settings=ApiSettings(allow_experimental_inference=True,service_key="synthetic-research-integration-not-secret-2026",
        enable_research_risk=True,enable_local_explanations=True,model_reference_dir=reference,
        model_manifest_sha256="ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac",
        research_risk_policy_path=ROOT/"policies/research_risk_v1_1/risk_policy.json",
        research_risk_policy_sha256="459b9b6da3f0a724df787606630942386928ddee8758439cb3da9131a30b075a")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health/ready").status_code==200
        headers={"X-Service-API-Key":settings.service_key.get_secret_value(),"X-Request-ID":"synthetic-research-real"}
        assert c.get("/internal/v2/risk-policy",headers=headers).json()["available"]
        yield c,headers


def test_all_bands_raw_parity_and_no_request_io(real_research_client,monkeypatch):
    import pandas as pd
    import joblib
    import psycopg
    c,headers=real_research_client
    def forbidden(*args,**kwargs):
        pytest.fail("No CSV/SQL/deserialization/fit/filesystem during inference")
    monkeypatch.setattr(psycopg,"connect",forbidden)
    monkeypatch.setattr(pd,"read_csv",forbidden)
    monkeypatch.setattr(joblib,"load",forbidden)
    cases=json.loads((ROOT/"examples/research_risk_synthetic_cases.json").read_text())
    for predictor in c.app.state.registry._predictors.values():
        for component in [predictor.pipeline,*predictor.pipeline.named_steps.values()]:
            for method in ["fit","fit_transform","partial_fit"]:
                if hasattr(component,method):monkeypatch.setattr(component,method,forbidden)
    monkeypatch.setattr(Path,"open",forbidden)
    monkeypatch.setattr(Path,"read_bytes",forbidden)
    monkeypatch.setattr(Path,"read_text",forbidden)
    covered={key:set() for key in cases[0]["expected_levels"]}
    for case in cases:
        for key,expected_level in case["expected_levels"].items():
            body={**case["payload"],"model_key":key}
            old=c.post("/internal/v1/predictions",json=body,headers=headers)
            new=c.post("/internal/v2/predictions",json=body,headers=headers)
            assert old.status_code==new.status_code==200
            data=new.json();probability=data["probability_no_show"]
            assert probability==pytest.approx(old.json()["probability_no_show"],abs=1e-10,rel=0)
            assert data["experimental_risk"]["level"]==expected_level
            assert data["operational_use_allowed"] is False and data["risk_level"] is None
            covered[key].add(expected_level)
        compare=c.post("/internal/v2/predictions/compare",json=case["payload"],headers=headers)
        assert compare.status_code==200
        assert [r["experimental_risk"]["level"] for r in compare.json()["results"]]==list(case["expected_levels"].values())
    assert all(levels=={"LOW","MEDIUM","HIGH"} for levels in covered.values())


def test_v1_shap_and_batch_with_research_enabled(real_research_client,payload):
    c,headers=real_research_client
    for key in ["random_forest_full_v1","xgboost_full_v1"]:
        r=c.post("/internal/v1/predictions/explain",json={**payload,"model_key":key},headers=headers)
        assert r.status_code==200
        assert r.json()["explanation"]["additivity_error"]<=(1e-7 if key.startswith("random") else 1e-5)
    batch={"mode":"compare","feature_version":payload["feature_version"],"feature_context":payload["feature_context"],
           "items":[{"features":payload["features"]}]*2}
    assert c.post("/internal/v1/predictions/batch",json=batch,headers=headers).status_code==200
