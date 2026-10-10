import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.model_info import MODEL_KEYS, ModelInfo
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features

TEST_KEY = "synthetic-unit-service-key-not-a-secret-2026"


class FakePredictor:
    def __init__(self, key, algorithm, probability):
        self.info = ModelInfo(key, algorithm, "a" * 64, "b" * 64, "c" * 64, "d" * 64, tuple(PREDICTOR_COLUMNS))
        self.probability = probability
        self.calls = []

    def predict_probability(self, features):
        self.calls.append(dict(features))
        if isinstance(self.probability, Exception):
            raise self.probability
        return self.probability


class FakeModelRegistry:
    def __init__(self):
        self.predictors = {MODEL_KEYS[0]: FakePredictor(MODEL_KEYS[0], "random_forest", .37),
                           MODEL_KEYS[1]: FakePredictor(MODEL_KEYS[1], "xgboost", .29)}

    @property
    def ready(self):
        return set(self.predictors) == set(MODEL_KEYS)

    def get(self, key):
        from depi_ml.domain.errors import UnknownModel
        if key not in self.predictors:
            raise UnknownModel
        return self.predictors[key]

    def list_models(self):
        return tuple(self.predictors[key].info for key in MODEL_KEYS if key in self.predictors)


@pytest.fixture
def registry():
    return FakeModelRegistry()


@pytest.fixture
def settings():
    return ApiSettings(allow_experimental_inference=True, service_key=TEST_KEY, expose_docs=True)


@pytest.fixture
def headers():
    return {"X-Service-API-Key": TEST_KEY, "X-Request-ID": "synthetic-request-001"}


@pytest.fixture
def payload():
    return {"feature_version": "BOOKING_BASELINE_V1", "feature_context": {"snapshot_type": "booking_creation",
            "builder_version": "synthetic_builder_v1"}, "features": synthetic_features()}


@pytest.fixture
def client(registry, settings):
    with TestClient(create_app(settings, registry_loader=lambda _: registry)) as client:
        yield client
