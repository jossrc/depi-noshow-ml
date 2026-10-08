"""Fixtures locales sintéticas compartidas: nunca consultan PostgreSQL real."""

import csv

import pytest


@pytest.fixture
def local_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    from depi_ml.analysis.dataset import digest, load_dataset, write_json
    from depi_ml.datasets.schema import EXPORT_COLUMNS
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for i in range(90):
        month = i // 30 + 1
        date = pd.Timestamp(f"2025-{month:02d}-05T12:00:00Z") + pd.Timedelta(hours=i % 30)
        appointment = date + pd.Timedelta(hours=1)
        local = appointment.tz_convert("America/Lima")
        row = {c: None for c in EXPORT_COLUMNS}
        row.update(appointment_id=10000 + i, client_id=20000 + i, prediction_at=date.isoformat(),
                   appointment_at=appointment.isoformat(), target=i % 2,
                   no_show_uid_minus_one="t" if i % 10 == 3 else "f", age_at_booking=20 + i % 40,
                   client_sex="F" if i % 2 else "M", booking_lead_days=1 / 24,
                   appointment_month=local.month, appointment_weekday=(local.dayofweek + 1) % 7,
                   appointment_hour=local.hour, duration_minutes=30 + i % 20, is_fwa="f",
                   clinic_id=1 + i % 3, scheduled_service_lines=1, distinct_body_areas=1,
                   single_body_area_id=5, has_medical_evaluation="f", has_type4_service="f",
                   previous_attended=0, previous_no_show=0, feature_version="SYNTHETIC_TEST_V1",
                   built_at="2025-04-01T00:00:00Z")
        records.append(row)
    path = source / "fixture.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    manifest = {"dataset_name": "synthetic_test", "row_count": len(records), "column_count": len(EXPORT_COLUMNS),
                "columns": EXPORT_COLUMNS, "size_bytes": path.stat().st_size, "sha256": digest(path),
                "source_schema": "analytics", "source_table": "appointment_training_dataset_v1",
                "feature_versions": ["SYNTHETIC_TEST_V1"], "exported_at": "2025-04-01T00:00:00Z"}
    write_json(source / "fixture_manifest.json", manifest)
    return load_dataset(path)


def write_label_fixture(path, dataset, records=None, snapshot="2025-04-02T00:00:00Z"):
    import pandas as pd
    from depi_ml.analysis.dataset import digest, write_json
    from depi_ml.datasets.label_availability import LABEL_COLUMNS, LABEL_DEFINITION, LABEL_SEMANTICS
    if records is None:
        records = [{"appointment_id": int(row.appointment_id), "target": int(row.target),
                    "outcome": "ATTENDED_COMPLETED" if row.target == 0 else "NO_SHOW",
                    "label_recorded_at": (row.appointment_at + pd.Timedelta(hours=2)).isoformat(),
                    "label_source": "laser_close_date" if row.target == 0 else "outcome_resolved_at"}
                   for row in dataset.frame.itertuples()]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    manifest = {"artifact_type": "label_availability", "format_version": 1, "columns": LABEL_COLUMNS,
                "column_count": len(LABEL_COLUMNS), "row_count": len(records), "sha256": digest(path),
                "size_bytes": path.stat().st_size, "source_dataset_sha256": dataset.sha256,
                "source_manifest_sha256": dataset.manifest_sha256,
                "source_dataset_exported_at": dataset.manifest["exported_at"],
                "source_snapshot_at": snapshot, "exported_at": snapshot,
                "definition": LABEL_DEFINITION, "semantics": LABEL_SEMANTICS}
    write_json(path.with_name(path.stem + "_manifest.json"), manifest)
    return records


@pytest.fixture
def labels(local_dataset, tmp_path):
    from depi_ml.datasets.label_availability import load_label_availability
    path = tmp_path / "fixture_labels.csv"
    write_label_fixture(path, local_dataset)
    return load_label_availability(local_dataset, path)
