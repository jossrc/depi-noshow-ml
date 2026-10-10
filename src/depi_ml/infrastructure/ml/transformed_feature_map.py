"""Mapa exhaustivo por metadatos ajustados; nunca divide nombres one-hot."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from depi_ml.datasets.schema import PREDICTOR_COLUMNS


class TransformedFeatureMap:
    def __init__(self, preprocess):
        names = preprocess.get_feature_names_out()
        assignments = [None] * len(names)
        seen = []
        for block, transformer, columns in preprocess.transformers_:
            if isinstance(transformer, str) and transformer == "drop":
                if len(columns):
                    raise ValueError("Unexpected dropped predictors")
                continue
            columns = list(columns)
            indices = preprocess.output_indices_[block]
            if not isinstance(indices, slice) or indices.step not in (None, 1):
                raise ValueError("Unsupported transformed indices")
            if isinstance(transformer, SimpleImputer):
                if list(transformer.get_feature_names_out(columns)) != columns:
                    raise ValueError("Imputation changed feature identity")
                expanded = columns
            elif isinstance(transformer, Pipeline):
                imputer = transformer.named_steps.get("imputer")
                encoder = transformer.named_steps.get("encoder")
                if not isinstance(imputer, SimpleImputer) or not isinstance(encoder, OneHotEncoder):
                    raise ValueError("Unsupported categorical transform")
                if list(imputer.get_feature_names_out(columns)) != columns:
                    raise ValueError("Imputation changed feature identity")
                # Original pipelines use no drop or infrequent grouping. Reject an
                # unfamiliar fitted topology rather than guess an attribution map.
                if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
                    raise ValueError("Unsupported one-hot grouping")
                if len(encoder.categories_) != len(columns):
                    raise ValueError("Invalid categorical metadata")
                expanded = [column for column, categories in zip(columns, encoder.categories_, strict=True)
                            for _ in categories]
                if len(encoder.get_feature_names_out(columns)) != len(expanded):
                    raise ValueError("Invalid one-hot width")
            else:
                raise ValueError("Unsupported transformer")
            if indices.stop - indices.start != len(expanded):
                raise ValueError("Invalid block width")
            for index, column in zip(range(indices.start, indices.stop), expanded, strict=True):
                if assignments[index] is not None or column not in PREDICTOR_COLUMNS:
                    raise ValueError("Duplicate or foreign transformed column")
                assignments[index] = column
            seen.extend(columns)
        if any(column is None for column in assignments) or sorted(seen) != sorted(PREDICTOR_COLUMNS):
            raise ValueError("Incomplete canonical feature mapping")
        self.assignments = tuple(assignments)

    def aggregate(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.assignments),) or not np.isfinite(values).all():
            raise ValueError("Invalid local contributions")
        totals = {name: [] for name in PREDICTOR_COLUMNS}
        for name, value in zip(self.assignments, values, strict=True):
            totals[name].append(float(value))
        from math import fsum
        grouped = tuple(fsum(totals[name]) for name in PREDICTOR_COLUMNS)
        if not np.isclose(fsum(grouped), fsum(values), rtol=0, atol=1e-10):
            raise ValueError("Grouping did not preserve SHAP sum")
        return grouped
