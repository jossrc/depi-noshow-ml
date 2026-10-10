"""Solo ML_* del entorno del proceso. No lee .env ni configuración de BD."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from depi_ml.domain.model_info import MODEL_KEYS


class ApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    env: Literal["development", "staging", "production"] = "development"
    mode: Literal["exploratory_demo"] = "exploratory_demo"
    allow_experimental_inference: bool = False
    service_key: SecretStr | None = Field(default=None, repr=False)
    model_reference_dir: Path | None = None
    model_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    default_model_key: Literal["random_forest_full_v1", "xgboost_full_v1"] = MODEL_KEYS[0]
    max_request_bytes: int = Field(default=262144, ge=1024, le=1048576)
    max_concurrent_predictions: int = Field(default=2, ge=1, le=16)
    expose_docs: bool = False

    @field_validator("service_key")
    @classmethod
    def strong_key(cls, value):
        if value is not None:
            key = value.get_secret_value()
            if len(key) < 32 or len(key) > 512 or not key.isascii() or not key.isprintable() or key.strip() != key:
                raise ValueError("ML_API_SERVICE_KEY requiere 32–512 caracteres ASCII sin espacios extremos.")
        return value

    @property
    def demo_enabled(self):
        return self.env == "development" and self.allow_experimental_inference and self.service_key is not None

    @classmethod
    def from_env(cls):
        names = {
            "env": "ML_API_ENV", "mode": "ML_API_MODE",
            "allow_experimental_inference": "ML_ALLOW_EXPERIMENTAL_INFERENCE",
            "service_key": "ML_API_SERVICE_KEY", "model_reference_dir": "ML_MODEL_REFERENCE_DIR",
            "model_manifest_sha256": "ML_MODEL_MANIFEST_SHA256", "default_model_key": "ML_DEFAULT_MODEL_KEY",
            "max_request_bytes": "ML_MAX_REQUEST_BYTES", "max_concurrent_predictions": "ML_MAX_CONCURRENT_PREDICTIONS",
            "expose_docs": "ML_EXPOSE_DOCS",
        }
        try:
            return cls(**{field: os.environ[name] for field, name in names.items() if name in os.environ})
        except ValueError:
            # Pydantic errors can echo input values, including secrets. Never propagate them.
            raise ValueError("Configuración ML_* inválida; consulta docs/api_operations.md.") from None
