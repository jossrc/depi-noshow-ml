"""Fixtures sintéticas pequeñas; no métricas de tesis ni conexiones a PostgreSQL."""

import csv
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from depi_ml.analysis.audit import REVIEW_REQUIREMENTS, VERIFIED_REQUIREMENTS, audit_dataset
from depi_ml.analysis.dataset import Phase2Error, digest, load_dataset, write_json
from depi_ml.analysis.reports import analyze
from depi_ml.datasets.schema import EXPORT_COLUMNS, PREDICTOR_COLUMNS
from depi_ml.evaluation.metrics import binary_metrics
from depi_ml.evaluation.reports import evaluate_partition
from depi_ml.training.preprocessing import FeatureContract, make_pipeline
from depi_ml.training.review import require_review
from depi_ml.training.splits import temporal_split


def approved_review(dataset, labels=None):
    return {"dataset_sha256": dataset.sha256, "feature_versions": dataset.manifest["feature_versions"],
            "reviewer": "Synthetic test only", "reviewed_at": "2025-04-02T00:00:00Z",
            "label_availability_sha256": labels.sha256 if labels else None,
            "label_availability_manifest_sha256": labels.manifest_sha256 if labels else None,
            "checks": {name: {"status": "verified" if name in VERIFIED_REQUIREMENTS else "acknowledged",
                              "evidence": "Synthetic fixture evidence for automated tests only."} for name in REVIEW_REQUIREMENTS}}


def split(frame):
    if "label_recorded_at" not in frame:
        # Marca registrada sintética explícita de esta fixture; no regla productiva de demora.
        frame = frame.assign(label_recorded_at=frame.appointment_at + pd.Timedelta(hours=2))
    return temporal_split(frame, "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z",
                          "2025-04-01T00:00:00Z", "2025-04-01T00:00:00Z")


def test_integrity_and_source_preserved(local_dataset, tmp_path):
    dataset = local_dataset
    assert not audit_dataset(dataset)["hard_blockers"]
    source_hash = digest(dataset.source)
    dataset.source.write_bytes(dataset.source.read_bytes() + b"\n")
    with pytest.raises(Phase2Error, match="SHA-256"):
        load_dataset(dataset.source)
    assert source_hash != digest(dataset.source)


@pytest.mark.parametrize("field,value", [("row_count", 91), ("size_bytes", 0), ("columns", list(reversed(EXPORT_COLUMNS))), ("feature_versions", ["wrong"])])
def test_manifest_contract(local_dataset, field, value):
    path = local_dataset.source.with_name("fixture_manifest.json")
    manifest = local_dataset.manifest.copy()
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(Phase2Error):
        load_dataset(local_dataset.source)


def test_short_csv_row_rejected_even_with_matching_hash(local_dataset):
    path = local_dataset.source
    with path.open("a", encoding="utf-8") as stream:
        stream.write("1,2,3\n")
    manifest = local_dataset.manifest.copy()
    manifest.update(sha256=digest(path), size_bytes=path.stat().st_size, row_count=91)
    write_json(path.with_name("fixture_manifest.json"), manifest)
    with pytest.raises(Phase2Error, match="cantidad de columnas"):
        load_dataset(path)


def test_audit_detects_history_and_future_errors(local_dataset):
    local_dataset.frame.loc[0, "prediction_at"] = local_dataset.frame.loc[0, "appointment_at"]
    local_dataset.frame.loc[1, "days_since_previous_no_show"] = -1
    local_dataset.frame.loc[2, "previous_no_show"] = 2
    local_dataset.frame.loc[2, "previous_no_show_rate"] = .1
    audit = audit_dataset(local_dataset)
    assert audit["csv_checks"]["prediction_not_before_appointment"] == 1
    assert audit["csv_checks"]["history_rate_inconsistent"] == 1
    assert audit["csv_checks"]["negative_days_since_previous_no_show"] == 1
    assert audit["hard_blockers"]


def test_observed_history_comparison_is_per_client_strictly_before_booking(local_dataset):
    df = local_dataset.frame
    df["client_id"] = 1 + df.index % 3
    df["previous_attended"] = 10
    df["previous_no_show"] = 10
    # Comparación de búsqueda vectorizada con referencia independiente por fila.
    comparable, exceeding = 0, 0
    for _, row in df.iterrows():
        for value, column in [(0, "previous_attended"), (1, "previous_no_show")]:
            count = int(((df.client_id == row.client_id) & (df.target == value) & (df.appointment_at < row.prediction_at)).sum())
            comparable += int(count > 0)
            exceeding += int(row[column] > count)
    audit = audit_dataset(local_dataset)
    assert audit["csv_checks"]["history_comparable_client_events"] == comparable
    assert audit["csv_checks"]["history_count_exceeds_observed_csv_events"] == exceeding


def test_no_dates_only_leakage_certificate(local_dataset, tmp_path):
    audit = audit_dataset(local_dataset)
    assert audit["training_status"].startswith("blocked")
    with pytest.raises(Phase2Error, match="Falta revisión"):
        require_review(local_dataset, audit, None)
    review = approved_review(local_dataset)
    review["checks"]["history_point_in_time"]["status"] = "acknowledged"
    path = tmp_path / "review.json"
    write_json(path, review)
    with pytest.raises(Phase2Error, match="history_point_in_time"):
        require_review(local_dataset, audit, path)
    review["checks"]["history_point_in_time"]["status"] = "verified"
    review["dataset_sha256"] = "wrong"
    write_json(path, review)
    with pytest.raises(Phase2Error, match="hash"):
        require_review(local_dataset, audit, path)


def test_temporal_split_purges_labels_including_long_lead(local_dataset):
    frame = local_dataset.frame.copy()
    frame.loc[0, "appointment_at"] = pd.Timestamp("2025-02-02T00:00:00Z")
    frame.loc[30, "appointment_at"] = pd.Timestamp("2025-03-02T00:00:00Z")
    parts = split(frame)
    assert 0 not in parts.train.index and 30 not in parts.validation.index
    assert parts.metadata["partitions"]["train"]["excluded_labels_not_available"] == 1
    for name, part in [("train", parts.train), ("validation", parts.validation), ("test", parts.test)]:
        cutoff = pd.Timestamp(parts.metadata["partitions"][name]["label_cutoff_exclusive"])
        assert (part.label_recorded_at < cutoff).all()
    assert parts.train.prediction_at.max() < parts.validation.prediction_at.min() < parts.test.prediction_at.min()
    assert not (set(parts.train.index) & set(parts.test.index))


def test_label_availability_exact_boundary_is_excluded(local_dataset):
    frame = local_dataset.frame.copy()
    frame.loc[0, "appointment_at"] = pd.Timestamp("2025-01-31T22:00:00Z")
    assert 0 not in split(frame).train.index


@pytest.mark.parametrize("value", [1, "2025-01-01", True, "invalid date"])
def test_invalid_label_recorded_type(local_dataset, value):
    frame = local_dataset.frame.assign(label_recorded_at=value)
    with pytest.raises(Phase2Error):
        split(frame)


def test_single_class_split_rejected(local_dataset):
    local_dataset.frame.loc[local_dataset.frame.prediction_at.dt.month == 3, "target"] = 0
    with pytest.raises(Phase2Error, match="sola clase"):
        split(local_dataset.frame)


def test_preprocessing_fit_only_on_train_and_safe_categories(local_dataset):
    parts = split(local_dataset.frame)
    train, test = parts.train.copy(), parts.test.copy()
    train["age_at_booking"] = 20.
    train.loc[train.index[:2], "age_at_booking"] = np.nan
    train["single_body_area_id"] = np.nan
    test["clinic_id"] = 999
    test["single_body_area_id"] = 12345
    test["age_at_booking"] = 110.
    pipeline = make_pipeline(DummyClassifier(strategy="prior"))
    pipeline.fit(train, train.target.astype(int))
    preprocess = pipeline.named_steps["preprocess"]
    numeric_names = preprocess.transformers_[0][2]
    age_statistic = preprocess.named_transformers_["numeric"].statistics_[numeric_names.index("age_at_booking")]
    assert age_statistic == 20
    before = preprocess.get_feature_names_out().copy()
    pipeline.predict_proba(test)
    assert np.array_equal(before, preprocess.get_feature_names_out())
    assert not any("999" in n or "12345" in n for n in before)
    assert not set(["client_id", "appointment_id", "target", "no_show_uid_minus_one", "prediction_at"]) & set(pipeline.named_steps["contract"].features_)
    numeric_categories = train.copy()
    numeric_categories["clinic_id"] = numeric_categories.clinic_id.astype(float)
    numeric_categories["is_fwa"] = False
    assert np.allclose(pipeline.predict_proba(train), pipeline.predict_proba(numeric_categories))


@pytest.mark.parametrize("feature", ["target", "client_id", "appointment_at", "no_show_uid_minus_one", "built_at", "label_recorded_at"])
def test_forbidden_predictors(local_dataset, feature):
    with pytest.raises(Phase2Error, match="ajenas"):
        FeatureContract([feature]).fit(local_dataset.frame)


def test_analysis_reports_no_client_ids_and_source_untouched(local_dataset, tmp_path):
    original = digest(local_dataset.source)
    # Cohorte pequeña solicitada, no expone citas individuales.
    frame = local_dataset.frame
    frame.loc[0, "client_id"] = 99467
    for column in ["no_show_uid_minus_one", "is_fwa", "has_medical_evaluation", "has_type4_service"]:
        frame[column] = frame[column].map({0: "f", 1: "t"})
    frame.to_csv(local_dataset.source, index=False)
    manifest = local_dataset.manifest.copy()
    manifest.update(sha256=digest(local_dataset.source), size_bytes=local_dataset.source.stat().st_size)
    write_json(local_dataset.source.with_name("fixture_manifest.json"), manifest)
    original = digest(local_dataset.source)
    out = analyze(local_dataset.source, None, tmp_path / "report")
    assert (out / "completion.json").is_file()
    special = json.loads((out / "atypical_cohort.json").read_text())
    assert special["present"] and special["rows"] == 1
    assert special["detailed_statistics_suppressed"]
    for path in out.iterdir():
        if path.suffix in {".csv", ".json", ".md"}:
            assert "99467" not in path.read_text()
            assert "20089" not in path.read_text()
    assert digest(local_dataset.source) == original
    with pytest.raises(Phase2Error, match="ya existe"):
        analyze(local_dataset.source, None, out)
    with pytest.raises(Phase2Error, match="fuera"):
        analyze(local_dataset.source, None, local_dataset.source.parent / "report")


def test_metrics_known_values():
    result = binary_metrics([0, 0, 1, 1], [.1, .2, .7, .8])
    assert result["roc_auc"] == 1
    assert result["pr_auc_average_precision"] == 1
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["brier_score"] == pytest.approx(.045)
    single = binary_metrics([0, 0], [.1, .2])
    assert single["roc_auc"] is None and single["pr_auc_average_precision"] is None


def test_reports_and_shap_with_real_estimators(local_dataset, tmp_path):
    pytest.importorskip("shap")
    from xgboost import XGBClassifier
    parts = split(local_dataset.frame)
    for name, estimator in [("rf", RandomForestClassifier(n_estimators=3, random_state=42)),
                            ("xgb", XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=42))]:
        pipeline = make_pipeline(estimator)
        pipeline.fit(parts.train, parts.train.target.astype(int))
        out = tmp_path / name
        metrics = evaluate_partition(pipeline, parts.test, out, PREDICTOR_COLUMNS, importance_samples=30, min_group=10, explain=True)
        assert metrics["rows"] == 30
        assert (out / "permutation_importance.csv").is_file()
        assert (out / "shap_importance.csv").is_file()


def test_train_evaluate_persistence_and_tamper_checks(local_dataset, labels, tmp_path, monkeypatch):
    from depi_ml.training import experiments
    from depi_ml.evaluation import runner
    # Las curvas y explicaciones se verifican aparte; aquí interesa el ciclo persistido completo.
    def small_report(pipeline, frame, output, features, threshold=.5, *args, **kwargs):
        output.mkdir(parents=True)
        return binary_metrics(frame.target, pipeline.predict_proba(frame[features])[:, 1], threshold)
    monkeypatch.setattr(experiments, "evaluate_partition", small_report)
    monkeypatch.setattr(runner, "evaluate_partition", small_report)
    source_hash = digest(local_dataset.source)
    review = tmp_path / "review.json"
    write_json(review, approved_review(local_dataset, labels))
    output = tmp_path / "experiment"
    with pytest.raises(Phase2Error):
        experiments.train(local_dataset.source, None, output, None, "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-04-01T00:00:00Z")
    assert not output.exists()
    experiments.train(local_dataset.source, None, output, review, "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-04-01T00:00:00Z", labels_path=labels.source)
    metadata = json.loads((output / "experiment.json").read_text())
    assert len(metadata["models"]) == 12
    assert metadata["test_status"] == "reserved_for_evaluate"
    assert metadata["artifact_format_version"] == 2
    assert metadata["label_availability"] == labels.identity()
    uid_entry = next(e for e in metadata["models"] if e["experiment"] == "without_uid_minus_one")
    assert uid_entry["split"]["partitions"]["train"]["rows"] == 27
    test_output = runner.evaluate(local_dataset.source, None, output, tmp_path / "evaluation", labels_path=labels.source)
    result = json.loads((test_output / "test_comparison.json").read_text())
    assert len(result["results"]) == 12
    from tests.conftest import write_label_fixture
    changed_labels = tmp_path / "different_labels.csv"
    records = labels.frame.to_dict("records")
    records[0]["label_recorded_at"] += pd.Timedelta(minutes=1)
    write_label_fixture(changed_labels, local_dataset, records)
    with pytest.raises(Phase2Error, match="difiere del utilizado"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "different_labels_eval", labels_path=changed_labels)
    assert not (tmp_path / "different_labels_eval").exists()
    copied_labels = tmp_path / "same_data_different_manifest.csv"
    copied_labels.write_bytes(labels.source.read_bytes())
    changed_manifest = labels.manifest.copy()
    changed_manifest["exported_at"] = "2025-04-02T00:01:00Z"
    write_json(copied_labels.with_name(copied_labels.stem + "_manifest.json"), changed_manifest)
    with pytest.raises(Phase2Error, match="difiere del utilizado"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "different_manifest_eval", labels_path=copied_labels)
    assert not (tmp_path / "different_manifest_eval").exists()
    legacy = metadata.copy()
    legacy["artifact_format_version"] = 1
    write_json(output / "experiment.json", legacy)
    with pytest.raises(Phase2Error, match="Versión de artefacto"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "legacy_eval", labels_path=labels.source)
    assert not (tmp_path / "legacy_eval").exists()
    write_json(output / "experiment.json", metadata)
    changed = metadata.copy()
    changed["source_code_sha256"] = "changed"
    write_json(output / "experiment.json", changed)
    with pytest.raises(Phase2Error, match="Código distinto"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "changed_code_eval", labels_path=labels.source)
    assert not (tmp_path / "changed_code_eval").exists()
    write_json(output / "experiment.json", metadata)
    model = output / metadata["models"][0]["model_path"]
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(Phase2Error, match="hash de modelo"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "tampered_eval", labels_path=labels.source)
    assert not (tmp_path / "tampered_eval").exists()
    assert digest(local_dataset.source) == source_hash
