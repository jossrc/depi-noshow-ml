"""Una única instantánea de lectura para validación y COPY."""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from depi_ml.config import ConfigurationError, Settings


@contextmanager
def read_only_connection(settings: Settings):
    with psycopg.connect(
        host=settings.host, port=settings.port, dbname=settings.database,
        user=settings.user, password=settings.password, connect_timeout=10,
        application_name="depi-ml-exporter", row_factory=dict_row, autocommit=True,
        options="-c default_transaction_read_only=on",
    ) as connection:
        with connection.transaction():
            # Antes de cualquier consulta y adquisición del snapshot.
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            connection.execute("SET LOCAL statement_timeout = '15000ms'")
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            connection.execute("SET LOCAL DateStyle TO 'ISO, YMD'")
            connection.execute("SET LOCAL extra_float_digits TO 3")
            identity = connection.execute("""
                SELECT current_database() AS database, current_user AS role,
                  current_setting('transaction_read_only') AS read_only,
                  current_setting('transaction_isolation') AS isolation
            """).fetchone()
            if (identity["database"] != settings.database or identity["role"] != settings.user
                    or identity["read_only"] != "on" or identity["isolation"] != "repeatable read"):
                raise ConfigurationError("La conexión exige identidad verificada y REPEATABLE READ READ ONLY.")
            yield connection
