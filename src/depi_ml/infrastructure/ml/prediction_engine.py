"""Invoca adaptadores con admisión acotada, sin fit ni filesystem por request."""

import logging
import math
from contextlib import contextmanager
from threading import BoundedSemaphore
from time import perf_counter

from depi_ml.application.dto.prediction_result import PredictionResult
from depi_ml.application.dto.explanation_result import ExplainedPrediction
from depi_ml.application.dto.batch_result import BatchResult
from depi_ml.application.ports.explainer import Explainer
from depi_ml.application.ports.model_registry import ModelRegistryPort
from depi_ml.domain.errors import BasePredictionException, ModelNotReady, PredictionEngineException, RateLimited
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.domain.errors import ExplanationNotReady, ExplanationFailed
from depi_ml.domain.risk_classifier import RiskClassifier
from depi_ml.infrastructure.telemetry.request_logging import request_id_context

logger = logging.getLogger(__name__)


class PredictionEngine:
    def __init__(self, registry: ModelRegistryPort, max_concurrent_predictions=2,
                 explainers: dict[str, Explainer] | None = None):
        self.registry = registry
        self._slots = BoundedSemaphore(max_concurrent_predictions)
        self._risk = RiskClassifier()
        self._explainers = dict(explainers or {})
        self._explanation_slots = BoundedSemaphore(1)

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

    def explain(self, model_key, features, top_k):
        with self._admit():
            self.registry.get(model_key)  # Unknown model remains 404, even with SHAP off.
            explainer = self._explainers.get(model_key)
            if explainer is None:
                raise ExplanationNotReady
            if not self._explanation_slots.acquire(blocking=False):
                raise RateLimited
            started = perf_counter()
            try:
                prediction = self._predict(model_key, features)
                try:
                    explanation = explainer.explain(features, prediction.probability_no_show, top_k)
                except Exception:
                    raise ExplanationFailed from None
                logger.info("event=explanation request_id=%s model_key=%s model_hash=%s elapsed_ms=%.3f",
                            request_id_context.get(), model_key, prediction.info.model_sha256[:12],
                            (perf_counter() - started) * 1000)
                return ExplainedPrediction(prediction, explanation)
            finally:
                self._explanation_slots.release()

    def batch(self, mode, items, model_key):
        with self._admit():
            keys = MODEL_KEYS if mode == "compare" else (model_key,)
            for key in keys:
                self.registry.get(key)  # Preflight before computing any row.
            started = perf_counter()
            results = []
            for features in items:
                if mode == "compare":
                    results.append(tuple(self._predict(key, features) for key in keys))
                else:
                    results.append(self._predict(model_key, features))
            logger.info("event=batch request_id=%s mode=%s batch_size=%s elapsed_ms=%.3f",
                        request_id_context.get(), mode, len(items), (perf_counter() - started) * 1000)
            return BatchResult(mode, tuple(results))
