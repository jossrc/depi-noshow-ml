import copy
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from depi_ml.api.main import create_app
from depi_ml.application.dto.explanation_result import ExplanationResult, ExplanationFactor
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.errors import RateLimited, PredictionEngineException
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.prediction_engine import PredictionEngine

PREFIX = "/internal/v1/predictions"

class FakeExplainer:
    def __init__(self, algorithm):
        self.algorithm, self.calls, self.failure = algorithm, [], None

    def explain(self, features, probability, top_k):
        self.calls.append(dict(features))
        if self.failure:
            raise self.failure
        xgb = self.algorithm == "xgboost"
        output = math.log(probability / (1 - probability)) if xgb else probability
        factor = ExplanationFactor("booking_lead_days", "neutral", 0., 0.)
        return ExplanationResult("log_odds" if xgb else "probability", "sigmoid" if xgb else "identity",
                                 output, output, output, 0., top_k, (factor,), 0.)

@pytest.fixture
def explained_client(registry, settings):
    explainers = {key: FakeExplainer(registry.get(key).info.algorithm) for key in MODEL_KEYS}
    app = create_app(settings.model_copy(update={"enable_local_explanations": True}),
                     registry_loader=lambda _: registry, explainer_loader=lambda _: explainers)
    with TestClient(app) as c:
        yield c, explainers
    assert app.state.explain_prediction is app.state.predict_batch is None

def batch_payload(payload, count=2, mode="single"):
    return {"feature_version": payload["feature_version"], "feature_context": payload["feature_context"],
            "mode": mode, "items": [{"features": copy.deepcopy(payload["features"])} for _ in range(count)]}

@pytest.mark.parametrize("key", [None, *MODEL_KEYS])
def test_explain_complete_parity_request_id(explained_client, registry, payload, headers, key):
    c, explainers = explained_client
    body = payload if key is None else {**payload, "model_key": key}
    single = c.post(PREFIX, json=body, headers=headers).json()
    r = c.post(PREFIX + "/explain", json=body, headers=headers)
    assert r.status_code == 200
    data = r.json()
    explanation = data.pop("explanation")
    assert data == single
    assert explanation["top_k"] == 5 and explanation["total_feature_count"] == 19
    assert explanation["explained_class"] == 1 and explanation["method"] == "tree_shap"
    assert explanation["explanation_scope"] == "local_association_not_causality"
    assert r.headers["X-Request-ID"] == headers["X-Request-ID"]
    actual = key or MODEL_KEYS[0]
    assert explainers[actual].calls == [payload["features"]]
    assert len(registry.get(actual).calls) == 2

@pytest.mark.parametrize("top_k,status", [(1, 200), (10, 200), (0, 422), (11, 422),
                                          (True, 422), (1., 422), ("5", 422), (None, 422)])
def test_explain_topk_strict(explained_client, payload, headers, top_k, status):
    c, explainers = explained_client
    assert c.post(PREFIX + "/explain", json={**payload, "top_k": top_k}, headers=headers).status_code == status
    if status == 422:
        assert all(not e.calls for e in explainers.values())

@pytest.mark.parametrize("failure", [False, True])
def test_optional_failure_keeps_prediction_ready(registry, settings, payload, headers, failure, caplog):
    def unavailable(_):
        raise ImportError("sensitive/path shap secret")
    options = {"explainer_loader": unavailable} if failure else {}
    with TestClient(create_app(settings.model_copy(update={"enable_local_explanations": failure}),
                               registry_loader=lambda _: registry, **options)) as c:
        r = c.post(PREFIX + "/explain", json=payload, headers=headers)
        assert r.status_code == 503 and r.json()["error"]["code"] == "EXPLANATION_NOT_READY"
        assert c.get("/health/ready").status_code == 200
        for path, body in [(PREFIX, payload), (PREFIX + "/compare", payload),
                           (PREFIX + "/batch", batch_payload(payload))]:
            assert c.post(path, json=body, headers=headers).status_code == 200
        assert c.post(PREFIX + "/explain", json={**payload, "model_key": "unknown"}, headers=headers).status_code == 404
    assert "sensitive" not in caplog.text + r.text

def test_explainer_failure_sanitized_slot_released(explained_client, payload, headers, caplog):
    c, explainers = explained_client
    e = explainers[MODEL_KEYS[0]]
    e.failure = ValueError("PII feature private_path key")
    r = c.post(PREFIX + "/explain", json=payload, headers=headers)
    assert r.status_code == 500 and r.json()["error"]["code"] == "EXPLANATION_FAILED"
    assert "PII" not in r.text + caplog.text
    e.failure = None
    assert c.post(PREFIX + "/explain", json=payload, headers=headers).status_code == 200

@pytest.mark.parametrize("mode", ["single", "compare"])
@pytest.mark.parametrize("count", [1, 2, 32])
def test_batch_full_nested_parity_order_flags(client, registry, payload, headers, mode, count):
    body = batch_payload(payload, count, mode)
    if count > 1:
        body["items"][1]["features"]["clinic_id"] = 999999
    r = client.post(PREFIX + "/batch", json=body, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == count and data["mode"] == mode and not data["operational_use_allowed"]
    assert data["request_id"] == r.headers["X-Request-ID"] == headers["X-Request-ID"]
    for i, item in enumerate(data["results"]):
        assert item["index"] == i
        single = client.post(PREFIX + ("/compare" if mode == "compare" else ""),
                             json={**payload, "features": body["items"][i]["features"]}, headers=headers)
        assert item["prediction"] == single.json()
    assert registry.get(MODEL_KEYS[0]).calls[:count] == [item["features"] for item in body["items"]]
    if mode == "compare":
        assert registry.get(MODEL_KEYS[1]).calls[:count] == registry.get(MODEL_KEYS[0]).calls[:count]

def test_batch_xgb_explicit(client, payload, headers):
    r = client.post(PREFIX + "/batch", json={**batch_payload(payload), "model_key": MODEL_KEYS[1]}, headers=headers)
    assert r.status_code == 200
    assert all(row["prediction"]["model_key"] == MODEL_KEYS[1] for row in r.json()["results"])

@pytest.mark.parametrize("count", [0, 33, 65])
def test_batch_limit_before_inference(client, registry, payload, headers, count):
    r = client.post(PREFIX + "/batch", json=batch_payload(payload, count), headers=headers)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_FEATURES"
    assert all(not p.calls for p in registry.predictors.values())

def test_batch_configured_limit(registry, settings, payload, headers):
    with TestClient(create_app(settings.model_copy(update={"max_batch_items": 64}), registry_loader=lambda _: registry)) as c:
        assert c.post(PREFIX + "/batch", json=batch_payload(payload, 64), headers=headers).status_code == 200
        assert c.post(PREFIX + "/batch", json=batch_payload(payload, 65), headers=headers).status_code == 422

@pytest.mark.parametrize("field", ["target", "client_id", "appointment_id", "phone", "email", "name"])
def test_batch_extra_atomic_sanitized(client, registry, payload, headers, field):
    body = batch_payload(payload, 4)
    body["items"][2]["features"][field] = "PII-secret-value"
    r = client.post(PREFIX + "/batch", json=body, headers=headers)
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == "items[2].features.[extra]"
    assert field not in r.text and "PII" not in r.text
    assert all(not p.calls for p in registry.predictors.values())

def test_invalid_middle_precise_location(client, registry, payload, headers):
    body = batch_payload(payload, 4)
    body["items"][3]["features"]["age_at_booking"] = 2
    r = client.post(PREFIX + "/batch", json=body, headers=headers)
    assert r.status_code == 422
    assert r.json()["error"]["details"] == [{"field": "items[3].features.age_at_booking", "reason": "greater_than_equal"}]
    assert all(not p.calls for p in registry.predictors.values())

@pytest.mark.parametrize("model_key", [None, MODEL_KEYS[0]])
def test_compare_prohibits_model_key(client, payload, headers, model_key):
    assert client.post(PREFIX + "/batch", json={**batch_payload(payload, mode="compare"), "model_key": model_key},
                       headers=headers).status_code == 422

@pytest.mark.parametrize("route", ["/explain", "/batch"])
@pytest.mark.parametrize("kind", ["version", "snapshot", "unknown", "inference"])
def test_new_errors(explained_client, registry, payload, headers, route, kind):
    c, _ = explained_client
    body = batch_payload(payload) if route == "/batch" else copy.deepcopy(payload)
    if kind == "version":
        body["feature_version"], expected = "OTHER", 409
    elif kind == "snapshot":
        body["feature_context"]["snapshot_type"], expected = "24h_before", 422
    elif kind == "unknown":
        body["model_key"], expected = "unknown", 404
    else:
        registry.get(MODEL_KEYS[0]).probability, expected = RuntimeError("private"), 500
    r = c.post(PREFIX + route, json=body, headers=headers)
    assert r.status_code == expected and "results" not in r.json() and "private" not in r.text

@pytest.mark.parametrize("route", ["/explain", "/batch"])
@pytest.mark.parametrize("auth", ["missing", "duplicate", "wrong"])
def test_auth_before_body_parse(client, headers, route, auth):
    key = headers["X-Service-API-Key"]
    pairs = [] if auth == "missing" else [("X-Service-API-Key", "wrong")] if auth == "wrong" else [
        ("X-Service-API-Key", key), ("X-Service-API-Key", key)]
    assert client.post(PREFIX + route, headers=pairs, content=b"invalid json").status_code == 401

@pytest.mark.parametrize("route", ["/explain", "/batch"])
@pytest.mark.parametrize("kind,expected", [("duplicate", 400), ("invalid", 400), ("utf8", 400),
                                           ("nan", 422), ("oversize", 413)])
def test_raw_body_guard(client, headers, route, kind, expected):
    raw = {"duplicate": b'{"mode":"single","mode":"single"}', "invalid": b"{", "utf8": b'"\xff"',
           "nan": b'{"age_at_booking":NaN}', "oversize": b" " * 262145}[kind]
    assert client.post(PREFIX + route, content=raw,
                       headers={**headers, "Content-Type": "application/json"}).status_code == expected

@pytest.mark.parametrize("setting,value", [("enable_local_explanations", "true"), ("enable_local_explanations", 1),
                                          ("max_batch_items", True), ("max_batch_items", "32"),
                                          ("max_batch_items", 0), ("max_batch_items", 65)])
def test_new_settings_strict(setting, value):
    with pytest.raises(ValidationError):
        ApiSettings(**{setting: value})

def test_new_env_flags_strict(monkeypatch):
    monkeypatch.setenv("ML_ENABLE_LOCAL_EXPLANATIONS", "true")
    monkeypatch.setenv("ML_MAX_BATCH_ITEMS", "64")
    s = ApiSettings.from_env()
    assert s.enable_local_explanations and s.max_batch_items == 64
    monkeypatch.setenv("ML_ENABLE_LOCAL_EXPLANATIONS", "1")
    with pytest.raises(ValueError):
        ApiSettings.from_env()

def test_one_slot_entire_batch_release_error(registry, payload):
    entered, release = Event(), Event()
    original = registry.get(MODEL_KEYS[0]).predict_probability
    def block(features):
        entered.set()
        assert release.wait(5)
        return original(features)
    registry.get(MODEL_KEYS[0]).predict_probability = block
    engine = PredictionEngine(registry, 1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.batch, "compare", (payload["features"],) * 4, MODEL_KEYS[0])
        try:
            assert entered.wait(5)
            for operation in (lambda: engine.predict(MODEL_KEYS[0], payload["features"]),
                              lambda: engine.compare(payload["features"]),
                              lambda: engine.explain(MODEL_KEYS[0], payload["features"], 5)):
                with pytest.raises(RateLimited):
                    operation()
        finally:
            release.set()
        assert len(future.result().predictions) == 4
    registry.get(MODEL_KEYS[0]).probability = ValueError("private")
    with pytest.raises(PredictionEngineException):
        engine.batch("single", (payload["features"],), MODEL_KEYS[0])
    registry.get(MODEL_KEYS[0]).probability = .37
    assert engine.predict(MODEL_KEYS[0], payload["features"]).probability_no_show == .37

def test_one_explanation_leaves_other_global_slot_available(registry, payload):
    entered, release = Event(), Event()
    e = FakeExplainer("random_forest")
    original = e.explain
    def block(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    e.explain = block
    engine = PredictionEngine(registry, 2, {MODEL_KEYS[0]: e})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.explain, MODEL_KEYS[0], payload["features"], 5)
        try:
            assert entered.wait(5)
            with pytest.raises(RateLimited):
                engine.explain(MODEL_KEYS[0], payload["features"], 5)
            assert engine.predict(MODEL_KEYS[1], payload["features"]).probability_no_show == .29
        finally:
            release.set()
        future.result()
    assert engine.explain(MODEL_KEYS[0], payload["features"], 5)

def test_http_429_shared_engine(client, payload, headers):
    engine = client.app.state.predict_batch.engine
    with engine._admit(), engine._admit():
        for suffix, body in [("/batch", batch_payload(payload)), ("/explain", payload)]:
            r = client.post(PREFIX + suffix, json=body, headers=headers)
            assert r.status_code == 429 and "probability_no_show" not in r.text


@pytest.mark.parametrize("route", ["/explain", "/batch"])
@pytest.mark.parametrize("column", PREDICTOR_COLUMNS)
def test_new_routes_require_all_19_keys(client, registry, payload, headers, route, column):
    del payload["features"][column]
    body = batch_payload(payload) if route == "/batch" else payload
    assert client.post(PREFIX + route, json=body, headers=headers).status_code == 422
    assert all(not p.calls for p in registry.predictors.values())


@pytest.mark.parametrize("route", ["/explain", "/batch"])
def test_new_routes_unready(registry, settings, payload, headers, route):
    with TestClient(create_app(settings.model_copy(update={"allow_experimental_inference": False}),
                               registry_loader=lambda _: pytest.fail("No load allowed"))) as c:
        r = c.post(PREFIX + route, json=batch_payload(payload) if route == "/batch" else payload, headers=headers)
        assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_READY"


def test_batch_mid_execution_failure_does_not_return_partial(client, registry, payload, headers):
    predictor = registry.get(MODEL_KEYS[0])
    original = predictor.predict_probability
    def fail_second(features):
        if len(predictor.calls) == 1:
            raise ValueError("private second item")
        return original(features)
    predictor.predict_probability = fail_second
    r = client.post(PREFIX + "/batch", json=batch_payload(payload, 4), headers=headers)
    assert r.status_code == 500 and r.json()["error"]["code"] == "INFERENCE_FAILED"
    assert "results" not in r.text and "private" not in r.text
    predictor.predict_probability = original
    assert client.post(PREFIX, json=payload, headers=headers).status_code == 200


def test_partial_explainer_availability_is_per_model(registry, settings, payload, headers):
    with TestClient(create_app(settings.model_copy(update={"enable_local_explanations": True}),
            registry_loader=lambda _: registry, explainer_loader=lambda _: {MODEL_KEYS[1]: FakeExplainer("xgboost")})) as c:
        assert c.get("/health/ready").status_code == 200
        assert c.post(PREFIX + "/explain", json=payload, headers=headers).status_code == 503
        assert c.post(PREFIX + "/explain", json={**payload, "model_key": MODEL_KEYS[1]}, headers=headers).status_code == 200


def test_engine_factory_is_injectable(registry, settings, payload, headers):
    from types import SimpleNamespace
    from depi_ml.application.dto.prediction_result import PredictionResult
    calls = []
    fake = SimpleNamespace(predict=lambda key, features: PredictionResult(registry.get(key).info, .12, None))
    def factory(*args):
        calls.append(args)
        return fake
    with TestClient(create_app(settings, registry_loader=lambda _: registry, engine_factory=factory)) as c:
        assert c.post(PREFIX, json=payload, headers=headers).json()["probability_no_show"] == .12
        assert c.app.state.explain_prediction.engine is fake
    assert len(calls) == 1 and all(not p.calls for p in registry.predictors.values())
