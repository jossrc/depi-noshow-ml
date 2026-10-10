"""Ejemplos públicos artificiales, sin claves ni datos de personas."""

from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features
from depi_ml.domain.model_info import MODEL_KEYS


def base_example():
    return {"feature_version": "BOOKING_BASELINE_V1",
            "feature_context": {"snapshot_type": "booking_creation", "builder_version": "synthetic_builder_v1"},
            "features": synthetic_features()}


def explanation_examples():
    return {name: {"summary": f"Artificial {name}", "value": {**base_example(), "model_key": key, "top_k": 5}}
            for name, key in zip(("rf", "xgb"), MODEL_KEYS, strict=True)}


def batch_examples():
    base = base_example()
    return {mode: {"summary": f"Artificial batch {mode}", "value": {
        "mode": mode, "feature_version": base["feature_version"], "feature_context": base["feature_context"],
        "items": [{"features": synthetic_features()}, {"features": {**synthetic_features(), "clinic_id": 999999}}],
        **({"model_key": MODEL_KEYS[0]} if mode == "single" else {})}} for mode in ("single", "compare")}
