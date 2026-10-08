"""Contrato compartido por entrenamiento y futura inferencia."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from depi_ml.analysis.dataset import Phase2Error
from depi_ml.datasets.schema import PREDICTOR_COLUMNS


CATEGORICAL_COLUMNS = {
    "client_sex", "clinic_id", "single_body_area_id", "is_fwa", "has_medical_evaluation",
    "has_type4_service", "appointment_month", "appointment_weekday", "appointment_hour",
}
CATALOG_COLUMNS = {"distinct_body_areas", "single_body_area_id", "has_medical_evaluation", "has_type4_service"}
BOOLEAN_COLUMNS = {"is_fwa", "has_medical_evaluation", "has_type4_service"}


def category_value(value, boolean=False):
    if pd.isna(value):
        return np.nan
    if boolean:
        mapping = {"t": "1", "true": "1", "1": "1", "1.0": "1",
                   "f": "0", "false": "0", "0": "0", "0.0": "0"}
        result = mapping.get(str(value).lower())
        if result is None:
            raise Phase2Error("Valor booleano inválido en el contrato predictor.")
        return result
    if isinstance(value, (int, float, np.number)):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


class FeatureContract(TransformerMixin, BaseEstimator):
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        features = list(PREDICTOR_COLUMNS if self.features is None else self.features)
        if not features or len(features) != len(set(features)) or not set(features) <= set(PREDICTOR_COLUMNS):
            raise Phase2Error("Variables ajenas o duplicadas en el contrato predictor.")
        self.features_ = features
        self.transform(X)
        self.feature_names_in_ = np.asarray(features, dtype=object)
        self.n_features_in_ = len(features)
        return self

    def transform(self, X):
        if not hasattr(X, "columns") or not set(self.features_) <= set(X.columns):
            raise Phase2Error("Faltan variables del contrato predictor.")
        data = X.loc[:, self.features_].copy()
        for column in self.features_:
            if column in CATEGORICAL_COLUMNS:
                data[column] = data[column].map(lambda value: category_value(value, column in BOOLEAN_COLUMNS))
            else:
                data[column] = data[column].astype(float)
        return data

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.features_, dtype=object)


def make_pipeline(estimator, features=None):
    features = list(PREDICTOR_COLUMNS if features is None else features)
    categorical = [c for c in features if c in CATEGORICAL_COLUMNS]
    numeric = [c for c in features if c not in CATEGORICAL_COLUMNS]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("category", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical),
    ], remainder="drop", sparse_threshold=1.0)
    return Pipeline([("contract", FeatureContract(features)), ("preprocess", preprocess), ("model", estimator)])
