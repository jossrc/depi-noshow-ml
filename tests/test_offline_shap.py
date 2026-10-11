"""Regresión SHAP con estimadores pequeños sintéticos, sin datos ni modelos privados."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
shap = pytest.importorskip("shap")
pytest.importorskip("sklearn")
xgb = pytest.importorskip("xgboost")

from depi_ml.evaluation.reports import shap_importance
from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features
from depi_ml.training.preprocessing import make_pipeline


def test_offline_xgb_shap_explains_sparse_model_not_zero_filled_model(tmp_path):
    frame = pd.DataFrame([dict(synthetic_features(), duration_minutes=40 if i % 3 == 0 else 0,
                              clinic_id=1 + i // 3 % 3) for i in range(60)])
    target = np.array([int(i % 3 == 0) for i in range(60)])
    pipeline = make_pipeline(xgb.XGBClassifier(n_estimators=8, max_depth=2, n_jobs=1, random_state=42))
    pipeline.fit(frame, target)
    sample = frame.iloc[:3]
    sparse = pipeline[:-1].transform(sample)
    assert hasattr(sparse, "tocoo")
    # Referencia independiente: ausentes CSR son missing; ceros almacenados siguen siendo ceros.
    coordinates = sparse.tocoo()
    dense = np.full(sparse.shape, np.nan)
    dense[coordinates.row, coordinates.col] = coordinates.data
    estimator = pipeline.named_steps["model"]
    np.testing.assert_allclose(estimator.predict_proba(dense), estimator.predict_proba(sparse))
    assert not np.allclose(estimator.predict_proba(sparse.toarray()), estimator.predict_proba(sparse))
    tree = shap.TreeExplainer(estimator)
    expected = np.asarray(tree.shap_values(dense))
    reconstructed = tree.expected_value + expected.sum(axis=1)
    np.testing.assert_allclose(reconstructed, estimator.predict(sparse, output_margin=True), atol=1e-5)
    shap_importance(pipeline, sample, tmp_path)
    result = pd.read_csv(tmp_path / "shap_importance.csv")
    np.testing.assert_allclose(result.mean_absolute_shap, np.abs(expected).mean(axis=0), atol=1e-7)
