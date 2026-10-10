"""Invoca adaptadores con admisión acotada, sin fit ni filesystem por request."""

import logging
import math
from contextlib import contextmanager
from threading import BoundedSemaphore
from time import perf_counter

from depi_ml.application.dto.prediction_result import PredictionResult
from depi_ml.application.ports.model_registry import ModelRegistryPort
from depi_ml.domain.errors import BasePredictionException, ModelNotReady, PredictionEngineException, RateLimited
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.domain.risk_classifier import RiskClassifier
from depi_ml.infrastructure.telemetry.request_logging import request_id_context

logger = logging.getLogger(__name__)


class PredictionEngine:
    def __init__(self, registry: ModelRegistryPort, max_concurrent_predictions=2):
        self.registry = registry
        self._slots = BoundedSemaphore(max_concurrent_predictions)
        self._risk = RiskClassifier()

    @contextmanager
    def _admit(self):
        if not self.registry.ready:
            raise ModelNotReady
        if not self._slots.acquire(blocking=False):
            raise RateLimited
        try:
            yield
        finally:
            self._slots.release()

    def _predict(self, key, features):
        predictor = self.registry.get(key)
        started = perf_counter()
        try:
            probability = float(predictor.predict_probability(features))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError
        except BasePredictionException:
            raise
        except Exception:
            raise PredictionEngineException from None
        logger.info('event=inference request_id=%s model_key=%s algorithm=%s model_hash=%s elapsed_ms=%.3f',
                    request_id_context.get(), key, predictor.info.algorithm, predictor.info.model_sha256[:12],
                    (perf_counter() - started) * 1000)
        return PredictionResult(predictor.info, probability, self._risk.classify(probability))

    def predict(self, model_key, features):
        with self._admit():
            return self._predict(model_key, features)

    def compare(self, features):
        with self._admit():
            return tuple(self._predict(key, features) for key in MODEL_KEYS)
