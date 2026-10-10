from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from depi_ml.application.dto.prediction_input import PredictionInput
from depi_ml.application.use_cases.predict_no_show import PredictNoShow
from depi_ml.application.use_cases.compare_models import CompareModels
from depi_ml.domain.errors import RateLimited, PredictionEngineException, FeatureVersionMismatch
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.infrastructure.ml.prediction_engine import PredictionEngine


@pytest.mark.parametrize("key", [None, *MODEL_KEYS])
def test_individual_selects_and_reuses(client, registry, payload, headers, key):
    if key:
        payload["model_key"] = key
    responses = [client.post("/internal/v1/predictions", json=payload, headers=headers) for _ in range(2)]
    assert all(r.status_code == 200 for r in responses)
    data = responses[0].json()
    actual = key or MODEL_KEYS[0]
    assert data["model_key"] == actual
    assert data["probability_no_show"] == registry.get(actual).probability
    assert responses[0].json() == responses[1].json()
    assert len(registry.get(actual).calls) == 2
    assert data["target_class"] == 1 and data["risk_level"] is None
    assert data["operational_use_allowed"] is False
    assert data["validation_status"] == "EXPLORATORY_NOT_VALIDATED"
    assert data["calibration_status"] == "not_operationally_validated"
    assert data["request_id"] == headers["X-Request-ID"]


def test_comparison_not_swapped_or_averaged(client, registry, payload, headers):
    registry.predictors = dict(reversed(list(registry.predictors.items())))
    r = client.post("/internal/v1/predictions/compare", json=payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert [s["model_key"] for s in data["results"]] == list(MODEL_KEYS)
    assert [s["algorithm"] for s in data["results"]] == ["random_forest", "xgboost"]
    assert [s["probability_no_show"] for s in data["results"]] == [.37, .29]
    assert data["comparison"]["absolute_probability_difference"] == pytest.approx(.08)
    assert data["comparison"]["ensemble_probability"] is None
    assert data["comparison"]["selected_winner"] is None
    assert data["risk_level"] is None
    assert data["operational_use_allowed"] is False
    a, b = (registry.get(k).calls for k in MODEL_KEYS)
    assert len(a) == len(b) == 1 and a == b == [payload["features"]]
    from depi_ml.datasets.schema import PREDICTOR_COLUMNS
    assert list(a[0]) == PREDICTOR_COLUMNS


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -.1, 1.1, RuntimeError("/sensitive/path secret")])
def test_prediction_failures_are_sanitized(client, registry, payload, headers, probability, caplog):
    registry.get(MODEL_KEYS[1]).probability = probability
    r = client.post("/internal/v1/predictions/compare", json=payload, headers=headers)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INFERENCE_FAILED"
    assert "results" not in r.json()
    assert "sensitive" not in r.text + caplog.text


def test_unknown_model(client, payload, headers):
    payload["model_key"] = "unregistered"
    assert client.post("/internal/v1/predictions", json=payload, headers=headers).status_code == 404


@pytest.mark.parametrize("route", ["", "/compare"])
def test_version_mismatch(client, payload, headers, route):
    payload["feature_version"] = "BOOKING_BASELINE_V2"
    r = client.post("/internal/v1/predictions" + route, json=payload, headers=headers)
    assert r.status_code == 409 and r.json()["error"]["code"] == "FEATURE_VERSION_MISMATCH"


def test_domain_dto_and_use_cases(registry, payload):
    dto = PredictionInput(payload["features"], payload["feature_version"], "booking_creation")
    engine = PredictionEngine(registry)
    assert PredictNoShow(engine, MODEL_KEYS[1]).execute(dto).probability_no_show == .29
    assert len(CompareModels(engine).execute(dto)) == 2
    with pytest.raises(TypeError):
        dto.features["clinic_id"] = 99
    with pytest.raises(FeatureVersionMismatch):
        PredictNoShow(engine).execute(PredictionInput(payload["features"], "other", "booking_creation"))


def test_concurrency_limit_is_real_and_released(registry, payload):
    entered, released = Event(), Event()
    def blocked(_):
        entered.set()
        assert released.wait(5)
        return .4
    registry.get(MODEL_KEYS[0]).predict_probability = blocked
    engine = PredictionEngine(registry, max_concurrent_predictions=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(engine.predict, MODEL_KEYS[0], payload["features"])
        try:
            assert entered.wait(5)
            with pytest.raises(RateLimited):
                engine.compare(payload["features"])
        finally:
            released.set()
        assert task.result().probability_no_show == .4
    registry.get(MODEL_KEYS[0]).predict_probability = lambda _: .5
    assert engine.predict(MODEL_KEYS[0], payload["features"]).probability_no_show == .5


def test_engine_releases_capacity_on_error(registry, payload):
    engine = PredictionEngine(registry, 1)
    registry.get(MODEL_KEYS[0]).probability = ValueError()
    with pytest.raises(PredictionEngineException):
        engine.compare(payload["features"])
    registry.get(MODEL_KEYS[0]).probability = .37
    assert len(engine.compare(payload["features"])) == 2
