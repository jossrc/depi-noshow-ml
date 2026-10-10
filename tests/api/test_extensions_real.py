"""Mandatory when originals exist and opt-in enabled; SHAP failures are failures."""
import math
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from depi_ml.api.main import create_app
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.infrastructure.ml.explanation_math import sigmoid

pytestmark = pytest.mark.skipif(os.environ.get("ML_RUN_REAL_INTEGRATION") != "1",
                               reason="Original models opt-in: ML_RUN_REAL_INTEGRATION=1")

@pytest.fixture(scope="module")
def real_extensions():
    root = Path(os.environ.get("ML_MODEL_REFERENCE_DIR", "experiments/phase2-exploratory-prompt5-20261008"))
    if not (root / "experiment.json").is_file():
        pytest.skip("Original local artifacts not mounted")
    settings = ApiSettings(allow_experimental_inference=True, enable_local_explanations=True,
                          service_key="synthetic-extension-integration-not-secret-2026",
                          model_reference_dir=root,
                          model_manifest_sha256="ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health/ready").status_code == 200
        assert set(c.app.state.explain_prediction.engine._explainers) == set(MODEL_KEYS), "SHAP must initialize BOTH originals"
        yield c, {"X-Service-API-Key": settings.service_key.get_secret_value(), "X-Request-ID": "synthetic-real-extension"}

@pytest.mark.parametrize("variant", ["standard", "unseen_clinic_nulls"])
def test_real_explain_and_batch_parity_math_no_io(real_extensions, payload, monkeypatch, variant):
    import numpy as np
    import pandas as pd
    import joblib
    import psycopg
    c, headers = real_extensions
    def forbidden(*args, **kwargs):
        pytest.fail("No fit/CSV/SQL/deserialization per HTTP request")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(psycopg, "connect", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    if variant != "standard":
        payload["features"].update(clinic_id=999999, age_at_booking=None, client_sex=None, is_fwa=None,
            has_type4_service=None, duration_minutes=None, distinct_body_areas=0, single_body_area_id=None,
            previous_attended=0, previous_no_show=0, previous_no_show_rate=None,
            days_since_previous_attended=None, days_since_previous_no_show=None)
    results = {}
    for key in MODEL_KEYS:
        predictor = c.app.state.registry.get(key)
        for component in [predictor.pipeline, *predictor.pipeline.named_steps.values(),
                          *predictor.pipeline.named_steps["preprocess"].named_transformers_.values()]:
            for method in ("fit", "fit_transform", "partial_fit"):
                if hasattr(component, method):
                    monkeypatch.setattr(component, method, forbidden)
        body = {**payload, "model_key": key}
        single = c.post("/internal/v1/predictions", json=body, headers=headers)
        response = c.post("/internal/v1/predictions/explain", json={**body, "top_k": 10}, headers=headers)
        assert response.status_code == 200, response.text
        full = response.json()
        e = full.pop("explanation")
        expected_single = single.json()
        assert full["probability_no_show"] == pytest.approx(expected_single["probability_no_show"], rel=0, abs=1e-10)
        assert {k: v for k, v in full.items() if k != "probability_no_show"} == {
            k: v for k, v in expected_single.items() if k != "probability_no_show"}
        probability = full["probability_no_show"]
        if variant == "standard":
            original = .6976400433344135 if key == MODEL_KEYS[0] else .9159286618232727
            assert probability == pytest.approx(original, rel=0, abs=1e-10)
        results[key] = full
        tolerance = 1e-7 if key == MODEL_KEYS[0] else 1e-5
        assert e["additivity_error"] <= tolerance
        assert e["reconstructed_output_in_explanation_space"] == pytest.approx(math.fsum((
            e["base_value"], *(f["contribution"] for f in e["factors"]), e["other_features_contribution"])), rel=0, abs=1e-12)
        reconstructed = e["reconstructed_output_in_explanation_space"]
        linked = reconstructed if key == MODEL_KEYS[0] else sigmoid(reconstructed)
        assert linked == pytest.approx(probability, rel=0, abs=1e-6)
        assert len(e["factors"]) == 10 and all(f["feature_name"] in PREDICTOR_COLUMNS for f in e["factors"])
        assert e["total_feature_count"] == 19
        assert [f["absolute_contribution"] for f in e["factors"]] == sorted(
            [f["absolute_contribution"] for f in e["factors"]], reverse=True)
        assert e["output_space"] == ("probability" if key == MODEL_KEYS[0] else "log_odds")
        again = c.post("/internal/v1/predictions", json=body, headers=headers)
        assert again.json()["probability_no_show"] == pytest.approx(probability, rel=0, abs=1e-10)
        batch = {"mode": "single", "model_key": key, "feature_version": payload["feature_version"],
                 "feature_context": payload["feature_context"], "items": [{"features": payload["features"]}] * 2}
        r = c.post("/internal/v1/predictions/batch", json=batch, headers=headers)
        assert r.status_code == 200
        assert all(row["prediction"]["probability_no_show"] == pytest.approx(probability, rel=0, abs=1e-10)
                   for row in r.json()["results"])
    batch.pop("model_key")
    batch["mode"] = "compare"
    r = c.post("/internal/v1/predictions/batch", json=batch, headers=headers)
    assert r.status_code == 200
    for row in r.json()["results"]:
        for prediction, key in zip(row["prediction"]["results"], MODEL_KEYS, strict=True):
            assert prediction["probability_no_show"] == pytest.approx(results[key]["probability_no_show"], rel=0, abs=1e-10)
            assert {k: v for k, v in prediction.items() if k != "probability_no_show"} == {
                k: v for k, v in results[key].items() if k != "probability_no_show"}

def test_real_changed_input_changes_genuine_local_factors(real_extensions, payload):
    c, headers = real_extensions
    changed = {**payload["features"], "booking_lead_days": 1., "previous_attended": 0,
               "previous_no_show": 3, "previous_no_show_rate": 1., "days_since_previous_attended": None}
    for key in MODEL_KEYS:
        a = c.post("/internal/v1/predictions/explain", json={**payload, "model_key": key}, headers=headers)
        b = c.post("/internal/v1/predictions/explain", json={**payload, "features": changed, "model_key": key}, headers=headers)
        assert a.status_code == b.status_code == 200
        assert a.json()["probability_no_show"] != b.json()["probability_no_show"]
        assert a.json()["explanation"]["factors"] != b.json()["explanation"]["factors"]
