import logging
import csv

import psycopg
import pytest

from depi_ml import cli


def test_help_is_executable(capsys):
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    assert "inspect-dataset" in capsys.readouterr().out


@pytest.mark.parametrize("size", ["0", "-1", "abc"])
def test_invalid_batch_size(size):
    with pytest.raises(SystemExit) as result:
        cli.main(["export-dataset", "--batch-size", size])
    assert result.value.code == 2


def test_database_error_does_not_expose_credentials(monkeypatch, caplog):
    monkeypatch.setenv("POSTGRES_PASSWORD", "fixture-password")
    def fail(*args, **kwargs):
        raise psycopg.OperationalError("password=fixture-password personal-data")
    monkeypatch.setattr(cli, "export_dataset", fail)
    with caplog.at_level(logging.ERROR):
        assert cli.main(["export-dataset"]) == 1
    assert "fixture-password" not in caplog.text
    assert "personal-data" not in caplog.text


def test_csv_errors_are_reported_without_row_contents(monkeypatch, caplog):
    monkeypatch.setenv("POSTGRES_PASSWORD", "fixture-password")
    def fail(*args, **kwargs):
        raise csv.Error("fixture personal-data")
    monkeypatch.setattr(cli, "export_dataset", fail)
    with caplog.at_level(logging.ERROR):
        assert cli.main(["export-dataset"]) == 1
    assert "personal-data" not in caplog.text
