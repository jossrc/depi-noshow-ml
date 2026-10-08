from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from depi_ml.config import Settings
from depi_ml.datasets.schema import COLUMNS, EXPORT_COLUMNS
from depi_ml.datasets.validator import (
    DatasetValidationError, inspect_dataset, report_json, validate_statistics,
)


def valid_stats():
    # Agregados de una fixture unitaria; no representan clientes de producción.
    return {
        "row_count": 2, "attended_count": 1, "no_show_count": 1,
        "invalid_target_count": 0, "duplicate_appointment_ids": 0,
        "missing_prediction_dates": 0, "invalid_temporal_count": 0,
        "nonfinite_timestamp_count": 0,
        "appointment_at_min": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "appointment_at_max": datetime(2025, 1, 2, tzinfo=timezone.utc),
        "unique_clients": 2, "distinct_clinics": 1, "no_show_uid_minus_one_count": 1,
        "negative_age_count": 0, "age_above_120_count": 0,
        "negative_booking_lead_count": 0, "nonfinite_booking_lead_count": 0,
        "null_counts": {name: 0 for name in EXPORT_COLUMNS},
    }


def mock_connection(stats=None):
    stats = deepcopy(stats or valid_stats())
    for name, count in stats.pop("null_counts").items():
        stats["null_" + name] = count
    columns = [{"attname": c.name, "typname": c.pg_type, "readable": True} for c in COLUMNS]
    connection = Mock()
    connection.execute.side_effect = [
        Mock(fetchone=lambda: {"oid": 10}), Mock(fetchone=lambda: {"allowed": True}),
        Mock(fetchone=lambda: {"oid": 20}), Mock(fetchall=lambda: columns),
        Mock(fetchone=lambda: stats),
        [{"feature_version": "unit-fixture-v1", "row_count": 2}],
    ]
    return connection


def test_report_class_counts_nulls_and_warnings():
    stats = valid_stats()
    stats["null_counts"]["age_at_booking"] = 1
    stats["negative_booking_lead_count"] = 1
    report = inspect_dataset(mock_connection(stats), Settings("localhost", 5432, "test", "test", "secret"))
    assert report["status"] == "passed"
    assert report["statistics"]["no_show_percentage"] == 50.0
    assert report["statistics"]["no_show_uid_minus_one_count"] == 1
    assert report["statistics"]["null_counts"]["age_at_booking"] == 1
    assert report["warnings"] == [{"check": "negative_booking_lead_count", "count": 1}]
    assert "2025-01-01T00:00:00+00:00" in report_json(report)
    assert "secret" not in report_json(report)


@pytest.mark.parametrize("field", [
    "invalid_target_count", "duplicate_appointment_ids", "invalid_temporal_count",
    "missing_prediction_dates", "nonfinite_timestamp_count",
])
def test_critical_anomalies(field):
    stats = valid_stats()
    stats[field] = 1
    assert validate_statistics(stats)
    with pytest.raises(DatasetValidationError) as failure:
        inspect_dataset(mock_connection(stats), Settings("localhost", 5432, "test", "test", "test"))
    assert failure.value.report["status"] == "failed"


def test_empty_and_null_identifiers():
    stats = valid_stats()
    stats["row_count"] = 0
    assert validate_statistics(stats)
    stats = valid_stats()
    stats["null_counts"]["appointment_id"] = 1
    assert validate_statistics(stats)


@pytest.mark.parametrize("stage, replacement", [
    (0, Mock(fetchone=lambda: None)),
    (1, Mock(fetchone=lambda: {"allowed": False})),
    (2, Mock(fetchone=lambda: None)),
    (3, Mock(fetchall=lambda: [{"attname": c.name, "typname": c.pg_type, "readable": False}
                             for c in COLUMNS])),
])
def test_missing_objects_and_permissions(stage, replacement):
    connection = mock_connection()
    responses = list(connection.execute.side_effect)
    responses[stage] = replacement
    connection.execute.side_effect = responses
    with pytest.raises(DatasetValidationError):
        inspect_dataset(connection, Settings("localhost", 5432, "test", "test", "test"))
