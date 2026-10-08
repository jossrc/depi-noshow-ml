"""Exploración explícita sin acreditar metodología ni utilizar prueba reservada."""

import csv
import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from depi_ml import cli
from depi_ml.analysis.audit import REVIEW_REQUIREMENTS, audit_dataset
from depi_ml.analysis.dataset import Phase2Error, digest, load_dataset, write_json
from depi_ml.datasets.label_availability import attach_labels, write_review_template
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.evaluation import runner
from depi_ml.evaluation.metrics import binary_metrics
from depi_ml.evaluation.reports import evaluate_partition
from depi_ml.training import experiments
from depi_ml.training.preprocessing import make_pipeline
from depi_ml.training.review import EXPLORATORY_WARNING, exploratory_review, require_review
from depi_ml.training.splits import temporal_split
from tests.conftest import write_label_fixture


CUTS = ("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z", "2025-04-01T00:00:00Z")


@pytest.mark.parametrize("use_template", [False, True])
def test_exploratory_trains_all_models_only_on_train_and_validation(
        local_dataset, labels, tmp_path, monkeypatch, use_template, capsys):
    review_path = tmp_path / "pending-review.json" if use_template else None
    if review_path:
        write_review_template(review_path, local_dataset, labels)
    paths = [local_dataset.source, local_dataset.manifest_source, labels.source,
             labels.source.with_name(labels.source.stem + "_manifest.json")]
    if review_path:
        paths.append(review_path)
    original_hashes = {path: digest(path) for path in paths}
    fit_calls, validation_calls = [], []
    original_fit = Pipeline.fit

    def checked_fit(self, X, y=None, *args, **kwargs):
        if "contract" in self.named_steps:
            assert set(X.index) <= set(range(30))
            assert set(X.columns) <= set(PREDICTOR_COLUMNS)
            assert X.index.equals(y.index)
            fit_calls.append(len(X))
        return original_fit(self, X, y, *args, **kwargs)

    def validation_report(pipeline, frame, output, features, threshold=.5, *args, report_context=None, **kwargs):
        assert set(frame.index) <= set(range(30, 60))
        validation_calls.append(len(frame))
        output.mkdir(parents=True)
        metrics = {**binary_metrics(frame.target, pipeline.predict_proba(frame[features])[:, 1], threshold),
                   **report_context}
        write_json(output / "metrics.json", metrics)
        return metrics

    def forbid(*args, **kwargs):
        raise AssertionError("Exploración no debe conectar a PostgreSQL ni ejecutar evaluate")

    monkeypatch.setattr(Pipeline, "fit", checked_fit)
    monkeypatch.setattr(experiments, "evaluate_partition", validation_report)
    monkeypatch.setattr(cli.Settings, "from_env", forbid)
    monkeypatch.setattr(cli, "read_only_connection", forbid)
    output = tmp_path / "exploratory"
    args = ["train", "--exploratory", "--csv", str(local_dataset.source), "--labels", str(labels.source),
            "--validation-start", CUTS[0], "--test-start", CUTS[1], "--test-end", CUTS[2],
            "--seed", "42", "--output", str(output)]
    if review_path:
        args += ["--review", str(review_path)]
    assert cli.main(args) == 0
    assert EXPLORATORY_WARNING in capsys.readouterr().out
    metadata = json.loads((output / "experiment.json").read_text())
    assert metadata["status"] == EXPLORATORY_WARNING
    assert metadata["training_mode"] == "exploratory"
    assert metadata["test_status"] == "reserved_exploratory_evaluation_forbidden"
    assert metadata["seed"] == 42
    assert set(metadata["methodological_review"]["pending_controls"]) == set(REVIEW_REQUIREMENTS)
    assert len(metadata["models"]) == len(fit_calls) == len(validation_calls) == 12
    assert {(entry["experiment"], entry["model"]) for entry in metadata["models"]} == {
        (name, model) for name in experiments.EXPERIMENTS for model in ["baseline", "random_forest", "xgboost"]}
    for entry in metadata["models"]:
        assert entry["split"]["test_partition_materialized"] is False
        assert entry["validation_metrics"]["status"] == EXPLORATORY_WARNING
        assert entry["validation_metrics"]["evaluation_partition"] == "validation"
        assert entry["hyperparameters"]["random_state"] == 42
        assert (output / entry["model_path"]).exists()
    assert (output / "validation_comparison.csv").exists()
    assert EXPLORATORY_WARNING in (output / "validation_comparison.md").read_text()
    assert not list(output.rglob("test_comparison.json"))
    assert not list(output.rglob("test"))
    snapshot = metadata["methodological_review"]["review_snapshot"]
    assert all(check["status"] == "pending" for check in snapshot["checks"].values())
    assert snapshot["reviewer"] == snapshot["reviewed_at"] == ""
    assert {path: digest(path) for path in paths} == original_hashes
    with pytest.raises(Phase2Error, match="EXPLORATORY_NOT_VALIDATED"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "forbidden-test", labels_path=labels.source)
    assert not (tmp_path / "forbidden-test").exists()
    with pytest.raises(Phase2Error):
        experiments.train(local_dataset.source, None, tmp_path / "definitive", review_path, *CUTS, labels_path=labels.source)
    assert not (tmp_path / "definitive").exists()


@pytest.mark.parametrize("failure", ["csv_hash", "labels_hash", "wrong_id", "wrong_target", "wrong_outcome",
                                     "attendance_at_prediction", "no_show_before_appointment", "future_snapshot",
                                     "negative_history", "bad_cuts", "immature_validation", "missing_labels"])
def test_exploratory_never_bypasses_technical_checks(local_dataset, labels, tmp_path, failure):
    records = labels.frame.to_dict("records")
    dataset = local_dataset
    if failure == "csv_hash":
        dataset.source.write_bytes(dataset.source.read_bytes() + b"\n")
    elif failure == "labels_hash":
        labels.source.write_bytes(labels.source.read_bytes() + b"\n")
    elif failure == "wrong_id":
        records[0]["appointment_id"] = 999999
    elif failure == "wrong_target":
        records[0]["target"] = 1
    elif failure == "wrong_outcome":
        records[0]["outcome"] = "NO_SHOW"
    elif failure == "attendance_at_prediction":
        records[0]["label_recorded_at"] = dataset.frame.iloc[0].prediction_at.isoformat()
    elif failure == "no_show_before_appointment":
        records[1]["label_recorded_at"] = (dataset.frame.iloc[1].prediction_at + pd.Timedelta(minutes=30)).isoformat()
    elif failure == "future_snapshot":
        records[0]["label_recorded_at"] = "2025-05-01T00:00:00Z"
    elif failure == "negative_history":
        with dataset.source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["previous_attended"] = "-1"
        with dataset.source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        manifest = dataset.manifest.copy()
        manifest.update(sha256=digest(dataset.source), size_bytes=dataset.source.stat().st_size)
        write_json(dataset.manifest_source, manifest)
        dataset = load_dataset(dataset.source)
    elif failure == "immature_validation":
        for index in range(30, 60):
            records[index]["label_recorded_at"] = CUTS[1]
    if failure not in {"csv_hash", "labels_hash"}:
        write_label_fixture(labels.source, dataset, records)
    cuts = (CUTS[1], CUTS[0], CUTS[2]) if failure == "bad_cuts" else CUTS
    output = tmp_path / "must-not-train"
    with pytest.raises(Phase2Error):
        experiments.train(dataset.source, None, output, None, *cuts, exploratory=True,
                          labels_path=None if failure == "missing_labels" else labels.source)
    assert not output.exists()


def test_exploratory_split_does_not_materialize_or_require_test_classes(local_dataset, labels):
    frame = attach_labels(local_dataset, labels)
    frame.loc[frame.prediction_at >= pd.Timestamp(CUTS[1]), "target"] = 0
    frame.loc[0, "label_recorded_at"] = pd.Timestamp(CUTS[0])
    frame.loc[30, "label_recorded_at"] = pd.Timestamp(CUTS[1])
    with pytest.raises(Phase2Error, match="sola clase"):
        temporal_split(frame, *CUTS, local_dataset.manifest["exported_at"])
    split = temporal_split(frame, *CUTS, local_dataset.manifest["exported_at"], include_test=False)
    assert split.test is None
    assert 0 not in split.train.index and 30 not in split.validation.index
    assert split.metadata["partitions"]["test"]["rows"] == 30
    assert split.metadata["partitions"]["test"]["status"] == "reserved_not_materialized"
    assert split.metadata["client_overlap_train_test"] is None


@pytest.mark.parametrize("flag,value", [("training_mode", "exploratory"), ("status", EXPLORATORY_WARNING),
                                      ("test_status", "reserved_exploratory_evaluation_forbidden")])
def test_evaluate_rejects_exploration_before_loading_models_or_splitting(local_dataset, tmp_path, monkeypatch, flag, value):
    experiment = tmp_path / "blocked-experiment"
    experiment.mkdir()
    write_json(experiment / "experiment.json", {flag: value})
    def forbid(*args, **kwargs):
        raise AssertionError("No debe deserializar, crear particiones ni calcular métricas de prueba")
    monkeypatch.setattr(runner.joblib, "load", forbid)
    monkeypatch.setattr(runner, "temporal_split", forbid)
    monkeypatch.setattr(runner, "evaluate_partition", forbid)
    with pytest.raises(Phase2Error, match=EXPLORATORY_WARNING):
        runner.evaluate(local_dataset.source, None, experiment, tmp_path / "test-output")
    assert not (tmp_path / "test-output").exists()


def test_exploratory_report_marks_metrics_tables_and_context(local_dataset, labels, tmp_path):
    split = temporal_split(attach_labels(local_dataset, labels), *CUTS, local_dataset.manifest["exported_at"], include_test=False)
    pipeline = make_pipeline(DummyClassifier(strategy="prior"))
    pipeline.fit(split.train, split.train.target.astype(int))
    pending = exploratory_review(local_dataset, audit_dataset(local_dataset), None, labels)
    context = {"status": EXPLORATORY_WARNING, "evaluation_partition": "validation",
               "pending_methodological_controls": pending["pending_controls"], "limitations": pending["limitations"]}
    output = tmp_path / "marked-report"
    metrics = evaluate_partition(pipeline, split.validation, output, PREDICTOR_COLUMNS, min_group=10,
                                 importance_samples=10, report_context=context)
    assert metrics["status"] == EXPLORATORY_WARNING
    assert json.loads((output / "report_context.json").read_text()) == context
    assert json.loads((output / "importance_metadata.json").read_text())["status"] == EXPLORATORY_WARNING
    for path in output.glob("*.csv"):
        assert pd.read_csv(path).status.eq(EXPLORATORY_WARNING).all()
    assert len(list(output.glob("*.png"))) == 4


def test_exploratory_review_does_not_accept_unbound_or_unsigned_claims(local_dataset, labels, tmp_path):
    path = tmp_path / "review.json"
    write_review_template(path, local_dataset, labels)
    snapshot = json.loads(path.read_text())
    snapshot["checks"]["history_point_in_time"] = {"status": "verified", "evidence": "Claim without signed human review evidence."}
    write_json(path, snapshot)
    summary = exploratory_review(local_dataset, audit_dataset(local_dataset), path, labels)
    assert "history_point_in_time" in summary["pending_controls"]
    assert summary["checks_automatically_verified"] is False
    assert summary["review_snapshot"] == snapshot
    with pytest.raises(Phase2Error):
        require_review(local_dataset, audit_dataset(local_dataset), path, labels)
    snapshot["label_availability_sha256"] = "wrong"
    write_json(path, snapshot)
    with pytest.raises(Phase2Error):
        exploratory_review(local_dataset, audit_dataset(local_dataset), path, labels)
