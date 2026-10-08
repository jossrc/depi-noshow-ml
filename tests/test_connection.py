from unittest.mock import MagicMock

import pytest

from depi_ml.config import Settings
from depi_ml.db import connection as module


@pytest.mark.parametrize("failure", [False, True])
def test_read_only_snapshot_and_transaction_lifecycle(monkeypatch, failure):
    connection = MagicMock()
    connection.__enter__.return_value = connection
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
