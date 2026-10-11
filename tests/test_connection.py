from unittest.mock import MagicMock

import pytest

from depi_ml.config import Settings
from depi_ml.db import connection as module


@pytest.mark.parametrize("failure", [False, True])
def test_read_only_snapshot_and_transaction_lifecycle(monkeypatch, failure):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = {
        "database": "test", "role": "test", "read_only": "on", "isolation": "repeatable read",
    }
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(module.psycopg, "connect", connect)
    settings = Settings("localhost", 5432, "test", "test", "test-only")
    if failure:
        with pytest.raises(RuntimeError):
            with module.read_only_connection(settings):
                raise RuntimeError("fixture failure")
    else:
        with module.read_only_connection(settings) as opened:
            assert opened is connection
    assert connect.call_args.kwargs["autocommit"] is True
    calls = [call.args[0] for call in connection.execute.call_args_list]
    assert calls[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    assert "SET LOCAL TIME ZONE 'UTC'" in calls
    transaction = connection.transaction.return_value
    exit_args = transaction.__exit__.call_args.args
    assert exit_args[0] is (RuntimeError if failure else None)


def test_read_only_session_default_timeout_and_verified_identity(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = {
        "database": "test", "role": "test", "read_only": "on", "isolation": "repeatable read",
    }
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(module.psycopg, "connect", connect)
    with module.read_only_connection(Settings("localhost", 5432, "test", "test", "fixture-secret")):
        pass
    assert connect.call_args.kwargs["options"] == "-c default_transaction_read_only=on"
    statements = [c.args[0] for c in connection.execute.call_args_list]
    assert "SET LOCAL statement_timeout = '15000ms'" in statements
    assert any("current_setting('transaction_read_only')" in s for s in statements)


@pytest.mark.parametrize("field,value", [("read_only", "off"), ("isolation", "read committed"),
                                         ("database", "unexpected"), ("role", "unexpected")])
def test_read_only_guard_refuses_unverified_session(monkeypatch, field, value):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    identity = {"database": "test", "role": "test", "read_only": "on", "isolation": "repeatable read"}
    identity[field] = value
    connection.execute.return_value.fetchone.return_value = identity
    monkeypatch.setattr(module.psycopg, "connect", MagicMock(return_value=connection))
    yielded = False
    with pytest.raises(ValueError, match="READ ONLY"):
        with module.read_only_connection(Settings("localhost", 5432, "test", "test", "fixture-secret")):
            yielded = True
    assert not yielded
