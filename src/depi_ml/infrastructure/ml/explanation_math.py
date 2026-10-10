"""Selección explícita de clase y comprobación de reconstrucción local."""

import math
import numpy as np

from depi_ml.application.dto.explanation_result import ExplanationFactor, ExplanationResult
from depi_ml.datasets.schema import PREDICTOR_COLUMNS

RF_ADDITIVITY_ATOL = 1e-7
XGB_ADDITIVITY_ATOL = 1e-5
LINK_ATOL = 1e-7


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def select_output(values, expected_value, width, algorithm, positive_index):
    """SHAP >=0.45 ndarray; legacy RF lists supported without guessing class."""
    base = np.asarray(expected_value, dtype=float)
    if algorithm == "random_forest":
        if positive_index not in (0, 1) or base.shape != (2,):
            raise ValueError("Invalid RF positive class/base")
        if isinstance(values, list):
            if len(values) != 2 or any(np.asarray(v).shape != (1, width) for v in values):
                raise ValueError("Invalid legacy RF SHAP shape")
            selected = np.asarray(values[positive_index], dtype=float)[0]
        else:
            array = np.asarray(values, dtype=float)
            if array.shape != (1, width, 2):
                raise ValueError("Ambiguous RF SHAP output")
            selected = array[0, :, positive_index]
        baseline = float(base[positive_index])
    elif algorithm == "xgboost":
        # Binary logistic XGB raw output always explains class 1. Reject a
        # contradictory class order rather than reinterpret the margin.
        array = np.asarray(values, dtype=float)
        if positive_index != 1 or array.shape != (1, width) or base.size != 1:
            raise ValueError("Invalid binary XGB SHAP output")
        selected, baseline = array[0], float(base.reshape(-1)[0])
    else:
        raise ValueError("Unsupported explanation algorithm")
    if not math.isfinite(baseline) or not np.isfinite(selected).all():
        raise ValueError("Nonfinite explanation")
    return selected, baseline


def build_explanation(contributions, base, output, probability, algorithm, top_k):
    if len(contributions) != len(PREDICTOR_COLUMNS) or type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("Invalid explanation contract")
    if not all(math.isfinite(value) for value in (*contributions, base, output, probability)):
        raise ValueError("Nonfinite explanation")
    if not 0 <= probability <= 1:
        raise ValueError("Invalid probability")
    reconstructed = math.fsum((base, *contributions))
    error = abs(reconstructed - output)
    if algorithm == "random_forest":
        space, link, tolerance, linked = "probability", "identity", RF_ADDITIVITY_ATOL, output
    elif algorithm == "xgboost":
        space, link, tolerance, linked = "log_odds", "sigmoid", XGB_ADDITIVITY_ATOL, sigmoid(output)
    else:
        raise ValueError("Unsupported explanation algorithm")
    if error > tolerance or abs(linked - probability) > LINK_ATOL:
        raise ValueError("Explanation does not reconstruct model output")
    ranked = sorted(enumerate(contributions), key=lambda pair: (-abs(pair[1]), pair[0]))
    factors = tuple(ExplanationFactor(PREDICTOR_COLUMNS[index],
                    "increases_model_output" if value > 0 else "decreases_model_output" if value < 0 else "neutral",
                    float(value), abs(float(value))) for index, value in ranked[:top_k])
    return ExplanationResult(space, link, float(base), float(output), reconstructed, error, top_k, factors,
                             math.fsum(value for _, value in ranked[top_k:]))
