from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import platform

import pytest

from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerifier, ArtifactVerificationError
from depi_ml.infrastructure.ml.model_registry import ModelRegistry


@pytest.fixture
def reference(tmp_path, monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-version")
    models = []
    for algorithm in ("random_forest", "xgboost"):
        rel = f"full/{algorithm}/model.joblib"
        path = tmp_path / rel
        path.parent.mkdir(parents=True)
        path.write_bytes(b"SYNTHETIC HASH FIXTURE; NEVER DESERIALIZE")
        models.append(dict(experiment="full", model=algorithm, features=list(PREDICTOR_COLUMNS),
                           model_path=rel, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           split={"test_partition_materialized": False}))
    m = dict(artifact_format_version=2, status="EXPLORATORY_NOT_VALIDATED", training_mode="exploratory",
             feature_versions=["BOOKING_BASELINE_V1"], predictor_contract=list(PREDICTOR_COLUMNS),
             modeling_partitions=["train", "validation"], test_status="reserved_exploratory_evaluation_forbidden",
             dataset_sha256="a" * 64, source_code_sha256="b" * 64, dataset_name="appointment_training_dataset_v1",
             dataset_rows=10, label_availability=dict(sha256="c" * 64, manifest_sha256="d" * 64,
             source_dataset_sha256="a" * 64, rows=10), versions={"python": platform.python_version(),
             **{p: "test-version" for p in ("scikit-learn", "xgboost", "numpy", "pandas", "joblib")}}, models=models)
    return tmp_path, m


def write_verifier(root, manifest):
    raw = json.dumps(manifest).encode()
    (root / "experiment.json").write_bytes(raw)
    return ArtifactVerifier(root, hashlib.sha256(raw).hexdigest())


def test_hashes_and_verified_snapshot(reference):
    root, manifest = reference
    verifier = write_verifier(root, manifest)
    artifacts = verifier.verify()
    assert [a.info.model_key for a in artifacts] == list(MODEL_KEYS)
    with verifier.verified_stream(artifacts[0]) as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == artifacts[0].info.model_sha256


@pytest.mark.parametrize("field,value", [
    ("status", "production"), ("training_mode", "reviewed"), ("artifact_format_version", 1),
    ("feature_versions", ["OTHER"]), ("predictor_contract", PREDICTOR_COLUMNS[:-1]),
    ("dataset_sha256", "invalid"), ("source_code_sha256", "invalid"),
    ("dataset_rows", 0), ("dataset_name", "other"), ("modeling_partitions", ["train", "validation", "test"]),
    ("test_status", "evaluated"), ("artifact_type", "exploratory_hyperparameter_round"),
])
def test_invalid_manifest(reference, field, value):
    root, manifest = reference
    manifest[field] = value
    with pytest.raises(ArtifactVerificationError):
        write_verifier(root, manifest).verify()


@pytest.mark.parametrize("package", ["python", "numpy", "pandas", "scikit-learn", "xgboost", "joblib"])
def test_incompatible_versions(reference, package):
    root, manifest = reference
    manifest["versions"][package] = "0.0.0"
    with pytest.raises(ArtifactVerificationError):
        write_verifier(root, manifest).verify()


@pytest.mark.parametrize("field,value", [("features", PREDICTOR_COLUMNS[:-1]), ("model_sha256", "0" * 64),
                    ("model_path", "../outside.joblib"), ("model_path", "/tmp/outside.joblib"),
                    ("experiment", "without_catalog"), ("model", "baseline")])
def test_invalid_model_entry(reference, field, value):
    root, manifest = reference
    manifest["models"][0][field] = value
    with pytest.raises(ArtifactVerificationError):
        write_verifier(root, manifest).verify()


def test_manifest_pin_detects_changes(reference):
    root, manifest = reference
    verifier = write_verifier(root, manifest)
    (root / "experiment.json").write_bytes(b"{}")
    with pytest.raises(ArtifactVerificationError) as error:
        verifier.verify()
    assert error.value.reason == "manifest_hash_mismatch"


@pytest.mark.parametrize("which", ["model", "manifest"])
def test_external_symlink_rejected(reference, tmp_path, which):
    root, manifest = reference
    verifier = write_verifier(root, manifest)
    # Outside the configured root, but still in the pytest temporary parent.
    external = root.parent / (root.name + "-external")
    external.write_bytes(b"outside")
    path = root / ("experiment.json" if which == "manifest" else manifest["models"][0]["model_path"])
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ArtifactVerificationError):
        verifier.verify()


def test_missing_duplicate_required_model(reference):
    root, manifest = reference
    manifest["models"].append(manifest["models"][0])
    with pytest.raises(ArtifactVerificationError):
        write_verifier(root, manifest).verify()


def test_change_between_verification_and_load(reference):
    root, manifest = reference
    verifier = write_verifier(root, manifest)
    artifact = verifier.verify()[0]
    artifact.path.write_bytes(b"changed")
    with pytest.raises(ArtifactVerificationError) as error:
        with verifier.verified_stream(artifact):
            pytest.fail("Must not deserialize changed bytes")
    assert error.value.reason == "model_changed_before_load"


def test_verify_both_hashes_before_any_load(reference, settings, monkeypatch):
    joblib = pytest.importorskip("joblib")
    root, manifest = reference
    verifier = write_verifier(root, manifest)
    (root / manifest["models"][1]["model_path"]).write_bytes(b"corrupted-xgb")
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("Must verify BOTH before joblib.load"))
    with pytest.raises(ArtifactVerificationError):
        ModelRegistry.load(settings.model_copy(update={"model_reference_dir": root,
                           "model_manifest_sha256": verifier.expected_manifest_sha256}))


def test_registry_is_immutable_and_deterministic(registry):
    real = ModelRegistry(reversed(list(registry.predictors.values())))
    assert real.ready
    assert [m.model_key for m in real.list_models()] == list(MODEL_KEYS)
    with pytest.raises(TypeError):
        real._predictors["other"] = None
    with pytest.raises(FrozenInstanceError):
        real._predictors = {}
    with pytest.raises(ArtifactVerificationError):
        ModelRegistry([registry.get(MODEL_KEYS[0])] * 2)
