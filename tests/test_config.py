from pathlib import Path

import pytest

from depi_ml.config import ConfigurationError, Settings


def test_defaults_and_password_redaction():
    settings = Settings.from_env({"POSTGRES_PASSWORD": "unit-test-secret"}, env_file=None)
    assert settings.port == 5432
    assert settings.schema == "analytics"
    assert settings.export_dir == Path("data/exports")
    assert "unit-test-secret" not in repr(settings)


@pytest.mark.parametrize("overrides", [
    {}, {"POSTGRES_PASSWORD": ""}, {"POSTGRES_PASSWORD": "CHANGE_ME"},
    {"POSTGRES_PORT": "bad"}, {"POSTGRES_PORT": "0"}, {"POSTGRES_PORT": "65536"},
    {"DATASET_BATCH_SIZE": "-1"}, {"DATASET_BATCH_SIZE": "1.5"},
    {"DATASET_SCHEMA": ""}, {"POSTGRES_HOST": " "},
])
def test_invalid_configuration(overrides):
    env = {"POSTGRES_PASSWORD": "test-only"} if "POSTGRES_PASSWORD" not in overrides and overrides else {}
    env.update(overrides)
    with pytest.raises(ConfigurationError):
        Settings.from_env(env, env_file=None)


def test_dotenv_and_environment_precedence(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_PASSWORD=test-only\nPOSTGRES_PORT=5544\n", encoding="utf-8")
    settings = Settings.from_env({"POSTGRES_PORT": "6543", "POSTGRES_DB": "test-db"}, env_file)
    assert settings.port == 6543
    assert settings.database == "test-db"
    assert Settings.from_env({}, env_file).port == 5544
