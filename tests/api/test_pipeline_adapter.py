import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.errors import PredictionEngineException
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError
from depi_ml.infrastructure.ml.sklearn_predictor import RandomForestPredictor, positive_class_index


class FakePipeline:
    classes_ = np.asarray([1, 0])

    def __init__(self, response=None):
        self.response = [[.8, .2]] if response is None else response
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame)
        return self.response

    def fit(self, *args):
        pytest.fail("fit forbidden on inference")

    partial_fit = fit
    fit_transform = fit


def test_positive_class_reversed_and_no_double_transform(registry, payload):
    pipeline = FakePipeline()
    predictor = RandomForestPredictor(registry.get(MODEL_KEYS[0]).info, pipeline)
    assert predictor.predict_probability(payload["features"]) == .8
    assert predictor.predict_probability(payload["features"]) == .8
    assert list(pipeline.calls[0].columns) == PREDICTOR_COLUMNS
    assert pipeline.calls[0].iloc[0]["is_fwa"] == False
    assert pipeline.calls[0].iloc[0]["clinic_id"] == 1
    assert len(pipeline.calls) == 2


@pytest.mark.parametrize("classes", [[0], [1, 2], [1, 1], ["0", "1"], [False, True], [[0, 1]]])
def test_bad_classes_blocked(classes):
    pipe = FakePipeline()
    pipe.classes_ = np.asarray(classes)
    with pytest.raises(ArtifactVerificationError):
        positive_class_index(pipe)


def test_predict_proba_required():
    pipe = FakePipeline()
    pipe.predict_proba = None
    with pytest.raises(ArtifactVerificationError):
        positive_class_index(pipe)


@pytest.mark.parametrize("probabilities", [[[float("nan"), .2]], [[float("inf"), 0]], [[1.1, -.1]], [[.3, .3]], [[.4]], [[.4, .6], [.4, .6]]])
def test_invalid_probability_outputs(registry, payload, probabilities):
    predictor = RandomForestPredictor(registry.get(MODEL_KEYS[0]).info, FakePipeline(probabilities))
    with pytest.raises(PredictionEngineException):
        predictor.predict_probability(payload["features"])
