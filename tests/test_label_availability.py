"""Correspondencia exacta, cierres tardíos y extracción sin escrituras SQL."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from unittest.mock import MagicMock

import pytest

pd = pytest.importorskip("pandas")

from depi_ml import cli
from depi_ml.analysis.dataset import Phase2Error, digest, write_json
from depi_ml.analysis.label_report import validate_labels
from depi_ml.analysis.audit import audit_dataset
from depi_ml.config import Settings
from depi_ml.datasets import label_exporter
from depi_ml.datasets.label_availability import attach_labels, load_label_availability, write_review_template
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.training.review import require_review
from depi_ml.training.splits import temporal_split
from tests.conftest import write_label_fixture
from tests.test_phase2 import approved_review


def parts(frame):
    return temporal_split(frame, "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z",
                          "2025-04-01T00:00:00Z", "2025-04-01T00:00:00Z")


def test_late_closes_and_original_export_cutoff(local_dataset, labels):
    records = labels.frame.to_dict("records")
    records[0]["label_recorded_at"] = "2025-03-15T00:00:00Z"
    records[30]["label_recorded_at"] = "2025-04-01T00:00:00Z"
    records[60]["label_recorded_at"] = "2025-04-01T00:00:00Z"
    write_label_fixture(labels.source, local_dataset, records)
    labels = load_label_availability(local_dataset, labels.source)
    result = parts(attach_labels(local_dataset, labels))
    for name, index in [("train", 0), ("validation", 30), ("test", 60)]:
        assert index not in getattr(result, name).index
        detail = result.metadata["partitions"][name]
        assert detail["excluded_recorded_at_or_after_cutoff"] == 1
        assert detail["excluded_labels_not_available"] == 1
        assert detail["excluded_by_target"]["0"]["at_or_after_cutoff"] == 1
        assert getattr(result, name).prediction_at.is_monotonic_increasing
    assert "label_delay_hours" not in result.metadata
    assert labels.summary["recorded_at_or_after_dataset_export"] == 2


def test_missing_labels_are_preserved_and_excluded(local_dataset, labels):
    records = labels.frame.to_dict("records")
    records[0]["label_recorded_at"] = ""
    write_label_fixture(labels.source, local_dataset, records)
    labels = load_label_availability(local_dataset, labels.source)
    assert len(labels.frame) == len(local_dataset.frame)
    assert labels.summary["missing_label_recorded_at"] == 1
    assert labels.summary["status"] == "validated_with_missing_labels"
    result = parts(attach_labels(local_dataset, labels))
    assert result.metadata["partitions"]["train"]["excluded_missing_label_recorded_at"] == 1
    assert 0 not in result.train.index


@pytest.mark.parametrize("value", ["2025-01-04T00:00:00Z", "2025-05-01T00:00:00Z"])
def test_inconsistent_dates_block_training_but_remain_auditable(local_dataset, labels, value, tmp_path):
    records = labels.frame.to_dict("records")
    records[0]["label_recorded_at"] = value
    write_label_fixture(labels.source, local_dataset, records)
    with pytest.raises(Phase2Error, match="Fechas inconsistentes"):
        load_label_availability(local_dataset, labels.source)
    out, report = validate_labels(local_dataset.source, None, labels.source, None, tmp_path / "audit",
                                 "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-04-01T00:00:00Z")
    assert report["validation"]["status"] == "blocked_inconsistent_dates"
    assert report["validation"]["inconsistent_dates"] == 1
    assert "temporal_split_preview" not in report
    assert (out / "label_availability_report.json").exists()


@pytest.mark.parametrize("value", ["2025-01-05T15:00:00", "infinity", "not-a-date"])
def test_invalid_timestamps_are_never_guessed(local_dataset, labels, value):
    records = labels.frame.to_dict("records")
    records[0]["label_recorded_at"] = value
    write_label_fixture(labels.source, local_dataset, records)
    with pytest.raises(Phase2Error):
        load_label_availability(local_dataset, labels.source)


@pytest.mark.parametrize("mutation", ["duplicate", "missing_id", "wrong_id", "target", "outcome", "source", "missing_row", "extra_row"])
def test_auxiliary_mismatch_rejected(local_dataset, labels, mutation):
    records = labels.frame.to_dict("records")
    if mutation == "duplicate":
        records[1]["appointment_id"] = records[0]["appointment_id"]
    elif mutation == "missing_id":
        records[0]["appointment_id"] = ""
    elif mutation == "wrong_id":
        records[0]["appointment_id"] = 999999
    elif mutation == "target":
        records[0]["target"] = 1
    elif mutation == "outcome":
        records[0]["outcome"] = "CANCELLED"
    elif mutation == "source":
        records[0]["label_source"] = "outcome_resolved_at"
    elif mutation == "missing_row":
        records.pop()
    else:
        records.append(records[0].copy())
    write_label_fixture(labels.source, local_dataset, records)
    with pytest.raises(Phase2Error):
        load_label_availability(local_dataset, labels.source)


@pytest.mark.parametrize("field,value", [
    ("source_dataset_sha256", "wrong"), ("source_manifest_sha256", "wrong"),
    ("sha256", "wrong"), ("row_count", 1), ("size_bytes", 1), ("columns", []),
    ("source_dataset_exported_at", "2025-03-31T00:00:00Z"), ("definition", {}),
])
def test_auxiliary_manifest_binding(local_dataset, labels, field, value):
    manifest = labels.manifest.copy()
    manifest[field] = value
    write_json(labels.source.with_name(labels.source.stem + "_manifest.json"), manifest)
    with pytest.raises(Phase2Error):
        load_label_availability(local_dataset, labels.source)


def test_unordered_auxiliary_joins_by_id_and_never_changes_source(local_dataset, labels):
    original = digest(local_dataset.source)
    manifest_hash = digest(local_dataset.manifest_source)
    write_label_fixture(labels.source, local_dataset, list(reversed(labels.frame.to_dict("records"))))
    labels = load_label_availability(local_dataset, labels.source)
    frame = attach_labels(local_dataset, labels)
    assert frame.appointment_id.tolist() == local_dataset.frame.appointment_id.tolist()
    expected = labels.frame.set_index("appointment_id").label_recorded_at
    assert frame.label_recorded_at.tolist() == expected.loc[frame.appointment_id].tolist()
    assert digest(local_dataset.source) == original
    assert digest(local_dataset.manifest_source) == manifest_hash
    assert "label_recorded_at" not in PREDICTOR_COLUMNS and len(PREDICTOR_COLUMNS) == 19


def test_missing_all_labels_gives_preview_but_prevents_training(local_dataset, labels):
    frame = attach_labels(local_dataset, labels)
    frame["label_recorded_at"] = pd.to_datetime([None] * len(frame), utc=True)
    with pytest.raises(Phase2Error, match="vacía"):
        parts(frame)
    preview = temporal_split(frame, "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-04-01T00:00:00Z",
                             "2025-04-01T00:00:00Z", require_both_classes=False)
    assert preview.train.empty
    assert preview.metadata["partitions"]["train"]["excluded_missing_label_recorded_at"] == 30


def test_review_remains_pending_and_binds_both_hashes(local_dataset, labels, tmp_path):
    path = tmp_path / "review.json"
    write_review_template(path, local_dataset, labels)
    pending = json.loads(path.read_text())
    assert all(check["status"] == "pending" for check in pending["checks"].values())
    assert pending["label_availability_sha256"] == labels.sha256
    with pytest.raises(Phase2Error):
        require_review(local_dataset, audit_dataset(local_dataset), path, labels)
    review = approved_review(local_dataset, labels)
    review["label_availability_manifest_sha256"] = "wrong"
    write_json(path, review)
    with pytest.raises(Phase2Error, match="hash del auxiliar"):
        require_review(local_dataset, audit_dataset(local_dataset), path, labels)


def test_local_validation_never_reads_postgres(local_dataset, labels, tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("No se permite acceder a PostgreSQL para validar archivos locales")
    monkeypatch.setattr(cli.Settings, "from_env", forbid)
    monkeypatch.setattr(label_exporter, "read_only_connection", forbid)
    output = tmp_path / "local-report"
    assert cli.main(["validate-label-availability", "--csv", str(local_dataset.source),
                     "--labels", str(labels.source), "--output", str(output)]) == 0
    assert (output / "methodology_review_template.json").is_file()


@pytest.fixture
def mocked_export(local_dataset, labels, monkeypatch):
    connection = MagicMock()
    transaction = {"snapshot_at": datetime(2025, 4, 2, tzinfo=timezone.utc), "read_only": "on", "isolation": "repeatable read"}
    column_types = [{"table_schema": name[0], "table_name": name[1], "column_name": name[2], "udt_name": kind}
                    for name, kind in label_exporter.REQUIRED_TYPES.items()]
    counts = {"missing_outcomes": 0, "outcome_identity_mismatch": 0, "outcome_type_mismatch": 0,
              "missing_attendance_session": 0, "session_identity_mismatch": 0}
    def execute(query):
        cursor = MagicMock()
        if "transaction_timestamp" in str(query):
            cursor.fetchone.return_value = transaction
        elif "information_schema" in str(query):
            cursor.fetchall.return_value = column_types
        else:
            cursor.fetchone.return_value = counts
        return cursor
    connection.execute.side_effect = execute
    @contextmanager
    def read_only(settings):
        yield connection
    monkeypatch.setattr(label_exporter, "read_only_connection", read_only)
    monkeypatch.setattr(label_exporter, "inspect_dataset", lambda *args: {})
    monkeypatch.setattr(label_exporter, "stream_copy", lambda c, s, p, b: p.write_bytes(local_dataset.source.read_bytes()))
    monkeypatch.setattr(label_exporter, "copy_labels", lambda c, p, b: p.write_bytes(labels.source.read_bytes()))
    return transaction, column_types, counts


def exporter_call(dataset, output):
    settings = Settings("localhost", 5432, "fixture", "fixture", "test-only")
    return label_exporter.export_label_availability(settings, dataset.source, None, output)


def test_export_checks_source_and_publishes_integrity(local_dataset, mocked_export, tmp_path):
    original = digest(local_dataset.source)
    output = tmp_path / "new_labels.csv"
    csv_path, manifest_path, quality_path, summary = exporter_call(local_dataset, output)
    assert summary["rows"] == 90
    manifest = json.loads(manifest_path.read_text())
    assert manifest["transaction"]["read_only"] is True
    assert manifest["source_dataset_matches_database"] is True
    assert manifest["source_dataset_sha256"] == original
    assert manifest["sha256"] == digest(csv_path)
    assert quality_path.exists()
    assert load_label_availability(local_dataset, csv_path).summary["status"] == "validated"
    assert digest(local_dataset.source) == original


@pytest.mark.parametrize("failure", ["database_drift", "writable", "wrong_timestamp_type", "missing_outcome", "duplicate_ids", "commit"])
def test_export_failures_do_not_publish_or_touch_dataset(local_dataset, labels, mocked_export, tmp_path, monkeypatch, failure):
    transaction, types, counts = mocked_export
    output = tmp_path / "not_published_labels.csv"
    original = digest(local_dataset.source)
    if failure == "database_drift":
        monkeypatch.setattr(label_exporter, "stream_copy", lambda c, s, p, b: p.write_bytes(local_dataset.source.read_bytes() + b"\n"))
    elif failure == "writable":
        transaction["read_only"] = "off"
    elif failure == "wrong_timestamp_type":
        types[-1]["udt_name"] = "timestamptz"
    elif failure == "missing_outcome":
        counts["missing_outcomes"] = 1
    elif failure == "duplicate_ids":
        records = labels.frame.to_dict("records")
        records[1]["appointment_id"] = records[0]["appointment_id"]
        write_label_fixture(labels.source, local_dataset, records)
    else:
        @contextmanager
        def fail_commit(settings):
            connection = MagicMock()
            connection.execute.return_value.fetchone.return_value = transaction
            yield connection
            raise OSError("Synthetic commit failure")
        monkeypatch.setattr(label_exporter, "read_only_connection", fail_commit)
        monkeypatch.setattr(label_exporter, "validate_sources", lambda *args: None)
    with pytest.raises((Phase2Error, OSError)):
        exporter_call(local_dataset, output)
    assert not output.exists()
    assert not output.with_name(output.stem + "_manifest.json").exists()
    assert not list(tmp_path.glob(".depi-labels-*"))
    assert digest(local_dataset.source) == original


def test_date_anomalies_export_for_audit_and_cli_returns_failure(local_dataset, labels, mocked_export, tmp_path, monkeypatch):
    records = labels.frame.to_dict("records")
    records[0]["label_recorded_at"] = "2025-01-04T00:00:00Z"
    write_label_fixture(labels.source, local_dataset, records)
    output = tmp_path / "audit_only_labels.csv"
    monkeypatch.setattr(cli.Settings, "from_env", lambda: Settings("localhost", 5432, "fixture", "fixture", "test-only"))
    assert cli.main(["export-label-availability", "--csv", str(local_dataset.source), "--output", str(output)]) == 1
    assert output.exists()
    manifest = json.loads(output.with_name(output.stem + "_manifest.json").read_text())
    assert manifest["validation"]["status"] == "blocked_inconsistent_dates"
    with pytest.raises(Phase2Error):
        load_label_availability(local_dataset, output)


def test_export_never_overwrites_original_or_prior_auxiliary(local_dataset, labels, mocked_export):
    original = digest(local_dataset.source)
    for output in [local_dataset.source, labels.source]:
        with pytest.raises(Phase2Error, match="No se sobrescriben"):
            exporter_call(local_dataset, output)
    assert digest(local_dataset.source) == original


def test_copy_query_uses_required_dates_without_writes():
    query = label_exporter.label_copy_query().as_string()
    assert "COPY (SELECT" in query and "TO STDOUT" in query
    assert 'l."LaserGCloseDate" AT TIME ZONE \'America/Lima\'' in query
    assert "ELSE o.outcome_resolved_at" in query
    assert "UPDATE " not in query and "INSERT " not in query and "COPY FROM" not in query
