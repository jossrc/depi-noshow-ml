from contextlib import contextmanager
import csv
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from depi_ml.config import Settings
from depi_ml.datasets import exporter
from depi_ml.datasets.schema import EXPORT_COLUMNS
from tests.test_dataset_validator import mock_connection


def csv_fixture():
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for target in (0, 1):
        row = {name: None for name in EXPORT_COLUMNS}
        row["target"] = target
        row["feature_version"] = 'unit,"fixture"\nwith newline'
        writer.writerow([row[name] for name in EXPORT_COLUMNS])
    return buffer.getvalue().encode("utf-8")


def test_verify_csv_counts_hash_nulls_and_multiline(tmp_path):
    path = tmp_path / "test.csv"
    content = csv_fixture()
    path.write_bytes(content)
    result = exporter.verify_csv(path, 2, 1)
    assert result["class_counts"] == {"0": 1, "1": 1}
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["size_bytes"] == len(content)


@pytest.mark.parametrize("content,count", [
    (b"wrong,header\n", 0), (csv_fixture(), 3),
    (csv_fixture().replace(b'\r\n', b'\r\nwrong\r\n', 1), 2),
])
def test_reject_invalid_csv(tmp_path, content, count):
    path = tmp_path / "test.csv"
    path.write_bytes(content)
    with pytest.raises(exporter.ExportError):
        exporter.verify_csv(path, count)


def test_reject_invalid_csv_target(tmp_path):
    path = tmp_path / "test.csv"
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(EXPORT_COLUMNS)
        row = [""] * len(EXPORT_COLUMNS)
        row[EXPORT_COLUMNS.index("target")] = "2"
        writer.writerow(row)
    with pytest.raises(exporter.ExportError, match="target"):
        exporter.verify_csv(path, 1)


def test_stream_copy_consumes_blocks(tmp_path):
    connection = MagicMock()
    connection.cursor.return_value.__enter__.return_value.copy.return_value.__enter__.return_value = [
        memoryview(b"first"), memoryview(b"second"),
    ]
    path = tmp_path / "test.csv"
    exporter.stream_copy(connection, Settings("localhost", 5432, "test", "test", "test"), path, 64)
    assert path.read_bytes() == b"firstsecond"


def setup_export(monkeypatch):
    @contextmanager
    def connection(_settings):
        yield mock_connection()
    monkeypatch.setattr(exporter, "read_only_connection", connection)
    monkeypatch.setattr(exporter, "stream_copy", lambda _c, _s, path, _size: path.write_bytes(csv_fixture()))


def test_complete_export_reports(tmp_path, monkeypatch):
    setup_export(monkeypatch)
    result = exporter.export_dataset(Settings("localhost", 5432, "test", "test", "test"), tmp_path / "custom.csv", 64)
    manifest = json.loads(result.manifest_path.read_text())
    quality = json.loads(result.quality_path.read_text())
    assert result.row_count == manifest["row_count"] == 2
    assert manifest["column_count"] == 27
    assert manifest["sha256"] == hashlib.sha256(result.csv_path.read_bytes()).hexdigest()
    assert quality["validations"]["csv_class_counts_match"]
    assert len(list(tmp_path.iterdir())) == 3


@pytest.mark.parametrize("failure", ["copy", "verify", "serialization"])
def test_failure_preserves_existing_outputs(tmp_path, monkeypatch, failure):
    setup_export(monkeypatch)
    destinations = [tmp_path / name for name in ("test.csv", "test_manifest.json", "test_quality_report.json")]
    for path in destinations:
        path.write_bytes(b"previous-valid-artifact")
    def fail(*args):
        if failure == "copy":
            args[2].write_bytes(b"partial")
        raise OSError("fixture failure")
    monkeypatch.setattr(exporter, {"copy": "stream_copy", "verify": "verify_csv", "serialization": "report_json"}[failure], fail)
    with pytest.raises(OSError):
        exporter.export_dataset(Settings("localhost", 5432, "test", "test", "test"), destinations[0])
    assert all(path.read_bytes() == b"previous-valid-artifact" for path in destinations)
    assert len(list(tmp_path.iterdir())) == 3


def test_publication_failure_rolls_back_all_outputs(tmp_path, monkeypatch):
    setup_export(monkeypatch)
    destinations = [tmp_path / name for name in ("test.csv", "test_manifest.json", "test_quality_report.json")]
    for path in destinations:
        path.write_bytes(b"previous")
    real_replace = exporter.os.replace
    def replace(source, destination):
        if Path(source).name == "dataset.csv":
            raise OSError("fixture publication failure")
        real_replace(source, destination)
    monkeypatch.setattr(exporter.os, "replace", replace)
    with pytest.raises(OSError):
        exporter.export_dataset(Settings("localhost", 5432, "test", "test", "test"), destinations[0])
    assert all(path.read_bytes() == b"previous" for path in destinations)
