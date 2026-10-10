"""Adaptadores C4: DataFrame crudo → pipeline completo → clase positiva 1."""

import math
from threading import Lock

import numpy as np
import pandas as pd

from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.errors import PredictionEngineException
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError


def positive_class_index(pipeline):
    classes = np.asarray(getattr(pipeline, "classes_", []))
    if (classes.shape != (2,) or classes.dtype.kind not in "iu"
            or set(classes.tolist()) != {0, 1} or not callable(getattr(pipeline, "predict_proba", None))):
        raise ArtifactVerificationError("invalid_positive_class_mapping")
    return int(np.flatnonzero(classes == 1)[0])


class SklearnPredictor:
    def __init__(self, info, pipeline):
        self.info, self.pipeline = info, pipeline
        self._positive_index = positive_class_index(pipeline)
        self._lock = Lock()

    def predict_probability(self, features):
        if set(features) != set(PREDICTOR_COLUMNS):
            raise PredictionEngineException
        frame = pd.DataFrame([{column: features[column] for column in PREDICTOR_COLUMNS}],
                             columns=PREDICTOR_COLUMNS)
        try:
            # Some estimators have mutable implementation caches: serialize access per model.
            with self._lock:
                probabilities = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
            if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not math.isclose(float(probabilities.sum()), 1, abs_tol=1e-6)):
                raise ValueError
            return float(probabilities[0, self._positive_index])
        except Exception:
            raise PredictionEngineException from None


class RandomForestPredictor(SklearnPredictor):
    """Pipeline Random Forest original full."""


class XGBoostPredictor(SklearnPredictor):
    """Pipeline XGBoost original full; nunca sustituido por RF."""
