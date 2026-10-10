"""Verificación de referencia confiable antes de cualquier joblib.load."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from tempfile import SpooledTemporaryFile

from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.errors import ModelNotReady
from depi_ml.domain.model_info import FEATURE_VERSION, VALIDATION_STATUS, ModelInfo
from depi_ml.infrastructure.ml.model_catalog import MODEL_CATALOG


class ArtifactVerificationError(ModelNotReady):
    def __init__(self, reason: str):
        super().__init__(self.message)
        self.reason = reason  # Código interno estático, nunca una ruta/input/secret.


@dataclass(frozen=True)
class VerifiedArtifact:
    path: Path
    info: ModelInfo


def sha256_file(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ArtifactVerifier:
    def __init__(self, reference_dir: Path, manifest_sha256: str):
        self.root = reference_dir.resolve(strict=True)
        self.expected_manifest_sha256 = manifest_sha256

    def _contained(self, relative: str):
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ArtifactVerificationError("invalid_artifact_path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ArtifactVerificationError("artifact_path_escape")
        resolved = (self.root / path).resolve(strict=True)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ArtifactVerificationError("artifact_path_escape")
        return resolved

    def verify(self) -> tuple[VerifiedArtifact, ...]:
        try:
            return self._verify()
        except ArtifactVerificationError:
            raise
        except Exception:
            raise ArtifactVerificationError("invalid_or_missing_reference") from None

    def _verify(self):
        path = self._contained("experiment.json")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ArtifactVerificationError("manifest_too_large")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.expected_manifest_sha256:
            raise ArtifactVerificationError("manifest_hash_mismatch")
        manifest = json.loads(raw)
        if (type(manifest["artifact_format_version"]) is not int or manifest["artifact_format_version"] != 2
                or manifest["status"] != VALIDATION_STATUS or manifest["training_mode"] != "exploratory"
                or manifest["feature_versions"] != [FEATURE_VERSION]
                or manifest["predictor_contract"] != PREDICTOR_COLUMNS
                or manifest["modeling_partitions"] != ["train", "validation"]
                or manifest["test_status"] != "reserved_exploratory_evaluation_forbidden"
                or manifest.get("artifact_type") is not None):
            raise ArtifactVerificationError("manifest_contract_mismatch")
        labels = manifest["label_availability"]
        identity_hashes = [manifest["dataset_sha256"], manifest["source_code_sha256"],
                           labels["sha256"], labels["manifest_sha256"]]
        if (any(not isinstance(h, str) or not re.fullmatch(r"[a-f0-9]{64}", h) for h in identity_hashes)
                or manifest["dataset_name"] != "appointment_training_dataset_v1"
                or type(manifest["dataset_rows"]) is not int or manifest["dataset_rows"] <= 0
                or labels["source_dataset_sha256"] != manifest["dataset_sha256"]
                or labels["rows"] != manifest["dataset_rows"]):
            raise ArtifactVerificationError("source_identity_mismatch")
        # Conservative policy: exact ML dependency versions; Python same major/minor.
        versions = manifest["versions"]
        if versions["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
            raise ArtifactVerificationError("python_version_mismatch")
        for package in ("scikit-learn", "xgboost", "numpy", "pandas", "joblib"):
            if versions[package] != importlib.metadata.version(package):
                raise ArtifactVerificationError("dependency_version_mismatch")
        artifacts = []
        for key, algorithm in MODEL_CATALOG.items():
            matches = [entry for entry in manifest["models"]
                       if entry["experiment"] == "full" and entry["model"] == algorithm]
            if len(matches) != 1:
                raise ArtifactVerificationError("required_model_missing_or_duplicate")
            entry = matches[0]
            if (entry["features"] != PREDICTOR_COLUMNS
                    or entry["model_path"] != f"full/{algorithm}/model.joblib"
                    or not re.fullmatch(r"[a-f0-9]{64}", entry["model_sha256"])
                    or entry["split"]["test_partition_materialized"] is not False):
                raise ArtifactVerificationError("model_contract_mismatch")
            model_path = self._contained(entry["model_path"])
            if sha256_file(model_path) != entry["model_sha256"]:
                raise ArtifactVerificationError("model_hash_mismatch")
            artifacts.append(VerifiedArtifact(model_path, ModelInfo(
                key, algorithm, entry["model_sha256"], digest, manifest["source_code_sha256"],
                manifest["dataset_sha256"], tuple(PREDICTOR_COLUMNS))))
        return tuple(artifacts)

    @contextmanager
    def verified_stream(self, artifact: VerifiedArtifact):
        # Deserialize the exact verified snapshot, avoiding path replacement races.
        # Spill >8 MiB to local temporary storage, not RAM proportional to large RF.
        with SpooledTemporaryFile(max_size=8 * 1024 * 1024) as snapshot:
            digest = hashlib.sha256()
            with artifact.path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    snapshot.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != artifact.info.model_sha256:
                raise ArtifactVerificationError("model_changed_before_load")
            snapshot.seek(0)
            yield snapshot
