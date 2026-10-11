"""Tree SHAP opcional sobre estimadores ajustados del pipeline original."""

import pandas as pd

from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.infrastructure.ml.explanation_math import build_explanation, select_output
from depi_ml.infrastructure.ml.sklearn_predictor import positive_class_index
from depi_ml.infrastructure.ml.shap_inputs import dense_tree_input
from depi_ml.infrastructure.ml.transformed_feature_map import TransformedFeatureMap

MAX_DENSE_ROW_BYTES = 1024 * 1024


class ShapExplainer:
    def __init__(self, predictor):
        # Deliberately deferred: importing API/CLI never requires SHAP.
        import shap
        self.predictor = predictor
        pipeline = predictor.pipeline
        self.positive_index = positive_class_index(pipeline)
        self.preprocess = pipeline.named_steps["preprocess"]
        self.mapping = TransformedFeatureMap(self.preprocess)
        if len(self.mapping.assignments) * 8 > MAX_DENSE_ROW_BYTES:
            raise ValueError("Dense row budget exceeded")
        self.estimator = pipeline.named_steps["model"]
        with predictor._lock:
            self.tree = shap.TreeExplainer(self.estimator, model_output="raw",
                                           feature_perturbation="tree_path_dependent")

    def explain(self, features, probability, top_k):
        if set(features) != set(PREDICTOR_COLUMNS):
            raise ValueError("Invalid feature contract")
        raw = pd.DataFrame([dict(features)], columns=PREDICTOR_COLUMNS)
        # One row only; same fitted contract and transformer, no fit/background.
        with self.predictor._lock:
            contracted = self.predictor.pipeline.named_steps["contract"].transform(raw)
            transformed = self.preprocess.transform(contracted)
            if transformed.shape != (1, len(self.mapping.assignments)):
                raise ValueError("Invalid transformed shape")
            dense = dense_tree_input(transformed, sparse_missing=self.predictor.info.algorithm == "xgboost")
            if dense.nbytes > MAX_DENSE_ROW_BYTES:
                raise ValueError("Dense row budget exceeded")
            values = self.tree.shap_values(dense, check_additivity=True)
            selected, base = select_output(values, self.tree.expected_value, dense.shape[1],
                                           self.predictor.info.algorithm, self.positive_index)
            if self.predictor.info.algorithm == "xgboost":
                output = float(self.estimator.predict(transformed, output_margin=True)[0])
            else:
                output = probability
            grouped = self.mapping.aggregate(selected)
            return build_explanation(grouped, base, output, probability, self.predictor.info.algorithm, top_k)


def load_explainers(registry):
    """Failure isolated per model; no second deserialization and no private data."""
    import logging
    result = {}
    for info in registry.list_models():
        try:
            result[info.model_key] = ShapExplainer(registry.get(info.model_key))
        except Exception:
            logging.getLogger(__name__).warning("event=explanation_unavailable model_key=%s reason=initialization_failed",
                                               info.model_key)
    return result
