"""Opt-in: modelos privados originales + fixtures artificiales, sin CSV/SQL/fit."""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError

pytestmark = pytest.mark.skipif(os.environ.get("ML_RUN_REAL_INTEGRATION") != "1",
                               reason="Modelos reales opt-in: ML_RUN_REAL_INTEGRATION=1")


@pytest.fixture(scope="module")
def real_client():
    root = Path(os.environ.get("ML_MODEL_REFERENCE_DIR", "experiments/phase2-exploratory-prompt5-20261008"))
    if not (root / "experiment.json").is_file():
        pytest.skip("Falta experimento original local")
    settings = ApiSettings(allow_experimental_inference=True,
                          service_key="synthetic-integration-key-not-a-secret-2026", model_reference_dir=root,
                          model_manifest_sha256=os.environ.get("ML_MODEL_MANIFEST_SHA256",
                              "ae7627340fe5207238da86f0db3d5ebc2b31250a1e4460c7926f169b5c658aac"))
    with TestClient(create_app(settings)) as c:
        assert c.get("/health/ready").status_code == 200, "Falló validación/carga de los dos modelos reales"
        yield c, settings


@pytest.mark.parametrize("variant", ["standard", "unseen_clinic_nulls"])
def test_real_http_matches_complete_pipeline(real_client, payload, monkeypatch, variant):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    import joblib
    import psycopg
    c, settings = real_client
    def forbidden(*args, **kwargs):
        pytest.fail("No SQL/CSV/joblib.load/fit allowed in requests")
    monkeypatch.setattr(psycopg, "connect", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    if variant == "unseen_clinic_nulls":
        payload["features"].update(clinic_id=999999, age_at_booking=None, client_sex=None, is_fwa=None,
            has_type4_service=None, duration_minutes=None, distinct_body_areas=0, single_body_area_id=None,
            previous_attended=0, previous_no_show=0, previous_no_show_rate=None,
            days_since_previous_attended=None, days_since_previous_no_show=None)
    frame = pd.DataFrame([payload["features"]], columns=PREDICTOR_COLUMNS)
    expected = []
    for key in MODEL_KEYS:
        predictor = c.app.state.registry.get(key)
        pipeline = predictor.pipeline
        for _, step in pipeline.steps:
            for name in ("fit", "fit_transform", "partial_fit"):
                if hasattr(step, name):
                    monkeypatch.setattr(step, name, forbidden)
        monkeypatch.setattr(pipeline, "fit", forbidden)
        expected.append(float(pipeline.predict_proba(frame)[0, int(np.flatnonzero(pipeline.classes_ == 1)[0])]))
    headers = {"X-Service-API-Key": settings.service_key.get_secret_value()}
    r = c.post("/internal/v1/predictions/compare", headers=headers, json=payload)
    assert r.status_code == 200
    assert [s["probability_no_show"] for s in r.json()["results"]] == pytest.approx(expected, abs=1e-12)
    assert r.json()["validation_status"] == "EXPLORATORY_NOT_VALIDATED"
    assert r.json()["comparison"]["ensemble_probability"] is None
    for key, probability in zip(MODEL_KEYS, expected, strict=True):
        single = c.post("/internal/v1/predictions", headers=headers, json={**payload, "model_key": key})
        assert single.status_code == 200
        assert single.json()["probability_no_show"] == pytest.approx(probability, abs=1e-12)


def test_copy_corrupted_real_reference_rejected_before_load(real_client, tmp_path, monkeypatch):
    import joblib
    _, settings = real_client
    source_manifest = settings.model_reference_dir / "experiment.json"
    manifest = json.loads(source_manifest.read_bytes())
    # Copy only metadata; corrupt a synthetic file at the same relative path. Originals untouched.
    raw = source_manifest.read_bytes()
    (tmp_path / "experiment.json").write_bytes(raw)
    entry = next(e for e in manifest["models"] if e["experiment"] == "full" and e["model"] == "random_forest")
    destination = tmp_path / entry["model_path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"intentionally-corrupted-copy-fixture")
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("Hash must fail before loading"))
    with pytest.raises(ArtifactVerificationError) as error:
        ModelRegistry.load(settings.model_copy(update={"model_reference_dir": tmp_path}))
    assert error.value.reason == "model_hash_mismatch"
