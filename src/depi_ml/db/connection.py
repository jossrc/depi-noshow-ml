"""Una única instantánea de lectura para validación y COPY."""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from depi_ml.config import Settings


@contextmanager
def read_only_connection(settings: Settings):
    with psycopg.connect(
        host=settings.host, port=settings.port, dbname=settings.database,
        user=settings.user, password=settings.password, connect_timeout=10,
        application_name="depi-ml-exporter", row_factory=dict_row, autocommit=True,
    ) as connection:
        with connection.transaction():
            # Antes de cualquier consulta y adquisición del snapshot.
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            connection.execute("SET LOCAL TIME ZONE 'UTC'")
            connection.execute("SET LOCAL DateStyle TO 'ISO, YMD'")
            connection.execute("SET LOCAL extra_float_digits TO 3")
            yield connection
