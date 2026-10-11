"""Representación densa que conserva la semántica de inferencia del árbol."""

import numpy as np


def dense_tree_input(transformed, *, sparse_missing=False):
    if sparse_missing and hasattr(transformed, "tocoo"):
        # XGBoost interpreta ausencias CSR como missing, incluso ceros omitidos.
        # Un cero explícitamente almacenado debe conservarse como cero.
        dense = np.full(transformed.shape, np.nan, dtype=float)
        coordinates = transformed.tocoo()
        dense[coordinates.row, coordinates.col] = coordinates.data
        return dense
    return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
