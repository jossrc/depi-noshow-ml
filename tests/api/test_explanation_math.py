import math
from types import SimpleNamespace
import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.infrastructure.ml.explanation_math import build_explanation, select_output, sigmoid
from depi_ml.infrastructure.ml.transformed_feature_map import TransformedFeatureMap
from depi_ml.infrastructure.ml.sklearn_predictor import positive_class_index
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError

@pytest.mark.parametrize("classes,index", [([0, 1], 1), ([1, 0], 0)])
@pytest.mark.parametrize("legacy", [False, True])
def test_rf_selects_fitted_positive_class(classes, index, legacy):
    pipeline = SimpleNamespace(classes_=np.array(classes), predict_proba=lambda _: None)
    assert positive_class_index(pipeline) == index
    values = np.array([[[.1, -.1], [.2, -.2]]])
    if legacy:
        values = [values[:, :, 0], values[:, :, 1]]
    selected, base = select_output(values, [.4, .6], 2, "random_forest", index)
    assert selected.tolist() == pytest.approx([.1, .2] if index == 0 else [-.1, -.2])
    assert base == [.4, .6][index]

@pytest.mark.parametrize("classes", [[0], [1], [1, 2], [0., 1.], ["0", "1"]])
def test_incompatible_classes_rejected(classes):
    with pytest.raises(ArtifactVerificationError) as error:
        positive_class_index(SimpleNamespace(classes_=np.array(classes), predict_proba=lambda _: None))
    assert error.value.reason == "invalid_positive_class_mapping"

def test_shapes_reject_ambiguous_rf_and_inverted_xgb():
    selected, base = select_output([[.1, .2]], .3, 2, "xgboost", 1)
    assert selected.tolist() == [.1, .2] and base == .3
    for algorithm, values, baseline, index in [("random_forest", [[.1, .2]], [.3, .7], 1),
           ("xgboost", [[[.1, .2], [.3, .4]]], .3, 1), ("xgboost", [[.1, .2]], .3, 0)]:
        with pytest.raises(ValueError):
            select_output(values, baseline, 2, algorithm, index)

@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_math_rejected(value):
    with pytest.raises(ValueError):
        select_output([[value]], 0, 1, "xgboost", 1)
    with pytest.raises(ValueError):
        build_explanation([value] + [0.] * 18, .5, .5, .5, "random_forest", 5)

def test_full_sum_topk_remainder_stable_ties_neutral():
    values = [.1, -.1, .04, .03] + [0.] * 15
    a = build_explanation(values, .5, .57, .57, "random_forest", 1)
    assert a.factors[0].feature_name == PREDICTOR_COLUMNS[0]
    assert a.other_features_contribution == pytest.approx(-.03)
    assert a.additivity_error < 1e-15
    assert a.base_value + a.factors[0].contribution != pytest.approx(.57)
    b = build_explanation(values, .5, .57, .57, "random_forest", 10)
    assert b.factors[1].direction == "decreases_model_output"
    assert all(f.direction == "neutral" for f in b.factors[4:])
    assert math.fsum((b.base_value, b.other_features_contribution,
                      *(f.contribution for f in b.factors))) == pytest.approx(.57)
    z = build_explanation([0.] * 19, .5, .5, .5, "random_forest", 5)
    assert [f.feature_name for f in z.factors] == PREDICTOR_COLUMNS[:5]
    assert all(f.direction == "neutral" for f in z.factors)

def test_logodds_link_and_reconstruction_failure():
    result = build_explanation([.2] * 19, -1., 2.8, sigmoid(2.8), "xgboost", 1)
    assert (result.output_space, result.output_link) == ("log_odds", "sigmoid")
    with pytest.raises(ValueError, match="reconstruct"):
        build_explanation([.2] * 19, -1., 2.9, sigmoid(2.9), "xgboost", 1)
    with pytest.raises(ValueError, match="reconstruct"):
        build_explanation([.2] * 19, -1., 2.8, .5, "xgboost", 5)

@pytest.fixture
def preprocess():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from depi_ml.training.preprocessing import make_pipeline
    from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features
    fixtures = [{**synthetic_features(), "client_sex": sex, "clinic_id": clinic,
                 "appointment_month": month, "single_body_area_id": area}
                for sex, clinic, month, area in [("F", 1, 1, 101), ("M", 2, 2, 102), ("X", 3, 3, 103)]]
    # Fitting only this minuscule synthetic unit fixture; no production artifact.
    pipeline = make_pipeline(RandomForestClassifier(n_estimators=2, random_state=42))
    pipeline.fit(pd.DataFrame(fixtures, columns=PREDICTOR_COLUMNS), [0, 1, 0])
    return pipeline.named_steps["preprocess"]

def test_exhaustive_onehot_mapping_with_underscores_and_sum(preprocess):
    mapping = TransformedFeatureMap(preprocess)
    assert set(mapping.assignments) == set(PREDICTOR_COLUMNS)
    assert all(mapping.assignments.count(name) == 3 for name in
               ("client_sex", "clinic_id", "appointment_month", "single_body_area_id"))
    values = np.arange(len(mapping.assignments), dtype=float) / 100
    grouped = mapping.aggregate(values)
    assert len(grouped) == 19
    assert math.fsum(grouped) == pytest.approx(math.fsum(values), abs=1e-12)
    for name in PREDICTOR_COLUMNS:
        assert grouped[PREDICTOR_COLUMNS.index(name)] == pytest.approx(math.fsum(
            value for column, value in zip(mapping.assignments, values) if column == name))

@pytest.mark.parametrize("corruption", ["overlap", "gap", "duplicate", "foreign", "drop"])
def test_invalid_mapping_rejected(preprocess, corruption):
    if corruption == "overlap":
        preprocess.output_indices_["category"] = slice(0, 2)
    elif corruption == "gap":
        preprocess.output_indices_["numeric"] = slice(1, 11)
    elif corruption == "drop":
        preprocess.transformers_.append(("extra", "drop", ["clinic_id"]))
    else:
        name, transformer, columns = preprocess.transformers_[0]
        columns = list(columns)
        columns[0] = "clinic_id" if corruption == "duplicate" else "private_field"
        preprocess.transformers_[0] = name, transformer, columns
    with pytest.raises(ValueError):
        TransformedFeatureMap(preprocess)
