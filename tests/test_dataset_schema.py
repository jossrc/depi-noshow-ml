import pytest

from depi_ml.config import Settings
from depi_ml.datasets.exporter import copy_query
from depi_ml.datasets.schema import COLUMNS, EXPORT_COLUMNS, PREDICTOR_COLUMNS
from depi_ml.datasets.validator import validate_columns


def test_feature_contract():
    assert len(PREDICTOR_COLUMNS) == len(set(PREDICTOR_COLUMNS)) == 19
    assert len(EXPORT_COLUMNS) == len(set(EXPORT_COLUMNS)) == 27
    excluded = {"appointment_id", "client_id", "target", "prediction_at", "appointment_at",
                "no_show_uid_minus_one", "feature_version", "built_at"}
    assert set(EXPORT_COLUMNS) - set(PREDICTOR_COLUMNS) == excluded


def test_required_columns_and_extra_columns():
    actual = {c.name: c.pg_type for c in COLUMNS}
    actual["unused_column"] = "text"
    assert validate_columns(actual) == []
    del actual["target"]
    assert "target" in validate_columns(actual)[0]


@pytest.mark.parametrize("column,bad_type", [
    ("target", "text"), ("prediction_at", "timestamp"),
    ("is_fwa", "int2"), ("booking_lead_days", "numeric"),
])
def test_incompatible_types(column, bad_type):
    actual = {c.name: c.pg_type for c in COLUMNS}
    actual[column] = bad_type
    assert validate_columns(actual)


def test_safe_explicit_unfiltered_copy():
    settings = Settings("localhost", 5432, "test", "test", "test", schema='schema"; DROP TABLE x; --')
    query = copy_query(settings).as_string()
    assert '"schema""; DROP TABLE x; --".' in query
    assert "SELECT *" not in query
    assert "LIMIT" not in query
    assert "WHERE" not in query
    assert "ORDER BY appointment_at, appointment_id" in query
    assert 'replace("prediction_at"::text' in query
