"""Ronda acotada: referencia intacta, cortes originales y solo validación."""

import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from depi_ml import cli
from depi_ml.analysis.dataset import Phase2Error, digest, write_json
from depi_ml.datasets.label_availability import attach_labels
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.evaluation import runner
from depi_ml.evaluation.metrics import binary_metrics
from depi_ml.training import optimization as opt
from depi_ml.training.experiments import EXPERIMENTS, json_parameters, source_hash, versions
from depi_ml.training.preprocessing import make_pipeline
from depi_ml.training.splits import temporal_split
from tests.conftest import write_label_fixture


@pytest.fixture
def reference(local_dataset, labels, tmp_path):
    import joblib
    XGBClassifier = pytest.importorskip("xgboost").XGBClassifier
    root = tmp_path / "original"
    root.mkdir()
    split = temporal_split(attach_labels(local_dataset, labels), "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z",
                           "2025-04-01T00:00:00Z", local_dataset.manifest["exported_at"], include_test=False)
    entries = []
    for experiment in opt.TARGET_EXPERIMENTS:
        features = EXPERIMENTS[experiment]
        for model, estimator in [
            ("baseline", DummyClassifier(strategy="prior", random_state=42)),
            ("random_forest", RandomForestClassifier(n_estimators=3, min_samples_leaf=2, random_state=42, n_jobs=1)),
            ("xgboost", XGBClassifier(n_estimators=3, max_depth=2, random_state=42, n_jobs=1, tree_method="hist")),
        ]:
            pipeline = make_pipeline(estimator, features)
            pipeline.fit(split.train[features], split.train.target.astype(int))
            path = root / experiment / model / "model.joblib"
            path.parent.mkdir(parents=True)
            joblib.dump(pipeline, path)
            entries.append({"experiment": experiment, "model": model, "features": features,
                            "model_path": str(path.relative_to(root)), "model_sha256": digest(path),
                            "hyperparameters": json_parameters(estimator.get_params()), "split": split.metadata,
                            "validation_metrics": {**binary_metrics(split.validation.target, pipeline.predict_proba(split.validation[features])[:, 1]),
                                                   "status": "EXPLORATORY_NOT_VALIDATED"}})
    write_json(root / "experiment.json", {
        "artifact_format_version": 2, "status": "EXPLORATORY_NOT_VALIDATED", "training_mode": "exploratory",
        "seed": 42, "threshold": .5, "dataset_sha256": local_dataset.sha256, "label_availability": labels.identity(),
        "predictor_contract": list(PREDICTOR_COLUMNS), "versions": versions(), "source_code_sha256": source_hash(), "models": entries})
    return root


def test_single_round_preserves_originals_and_never_models_test(local_dataset, labels, reference, tmp_path, monkeypatch):
    monkeypatch.setattr(opt, "CONFIGURATIONS", {"random_forest": {"rf_small": {"min_samples_leaf": 3}},
                                              "xgboost": {"xgb_small": {"max_depth": 3}}})
    artifacts = [*reference.rglob("*"), local_dataset.source, local_dataset.manifest_source, labels.source,
                 labels.source.with_name(labels.source.stem + "_manifest.json")]
    protected = {p: digest(p) for p in artifacts if p.is_file()}
    fit, predict = Pipeline.fit, Pipeline.predict_proba
    fit_rows, prediction_rows = [], []
    def train_only(self, X, y=None, *args, **kwargs):
        if "contract" in self.named_steps:
            assert set(X.index) <= set(range(30))
            fit_rows.append(len(X))
        return fit(self, X, y, *args, **kwargs)
    def validation_only(self, X, *args, **kwargs):
        assert set(X.index) <= set(range(30, 60))
        prediction_rows.append(len(X))
        return predict(self, X, *args, **kwargs)
    def forbid(*args, **kwargs):
        raise AssertionError("La ronda no debe acceder a PostgreSQL")
    monkeypatch.setattr(Pipeline, "fit", train_only)
    monkeypatch.setattr(Pipeline, "predict_proba", validation_only)
    monkeypatch.setattr(cli.Settings, "from_env", forbid)
    monkeypatch.setattr(cli, "read_only_connection", forbid)
    output = tmp_path / "round"
    assert cli.main(["optimize-exploratory", "--csv", str(local_dataset.source), "--labels", str(labels.source),
                     "--reference", str(reference), "--output", str(output)]) == 0
    metadata = json.loads((output / "experiment.json").read_text())
    assert metadata["rounds"] == 1
    assert metadata["threshold"] == .5 and metadata["seed"] == 42
    assert metadata["status"] == "EXPLORATORY_NOT_VALIDATED"
    assert metadata["definitive_model"] is None
    assert metadata["test_status"] == "reserved_exploratory_evaluation_forbidden"
    assert metadata["reference_validation_metrics_reproduced"] is True
    assert len(metadata["models"]) == 10  # 6 referencias y 4 candidatos, sin refit originales.
    assert len(fit_rows) == 4 and len(prediction_rows) == 10
    for entry in metadata["models"]:
        assert entry["split"]["test_partition_materialized"] is False
        assert entry["hyperparameters"]["random_state"] == 42
        assert entry["validation_metrics"]["threshold"] == .5
        assert [m["month"] for m in entry["monthly_metrics"]] == ["2025-02"]
    assert {p: digest(p) for p in protected} == protected
    assert not list(output.rglob("test_comparison.json"))
    assert (output / "validation_monthly.csv").exists()
    assert (output / "optimization_report.md").exists()
    assert len(list(output.glob("monthly_*.png"))) == 4
    with pytest.raises(Phase2Error, match="EXPLORATORY_NOT_VALIDATED"):
        runner.evaluate(local_dataset.source, None, output, tmp_path / "test-eval", labels_path=labels.source)
    with pytest.raises(Phase2Error, match="ya existe"):
        opt.optimize(local_dataset.source, None, labels.source, None, reference, output)


@pytest.mark.parametrize("failure", ["model_hash", "metadata_hash", "missing_model", "features", "versions",
                                     "seed", "threshold", "split", "stored_metrics", "labels_hash", "critical_date"])
def test_invalid_reference_or_labels_block_before_any_training(local_dataset, labels, reference, tmp_path, monkeypatch, failure):
    path = reference / "experiment.json"
    metadata = json.loads(path.read_text())
    if failure == "model_hash":
        model = reference / metadata["models"][0]["model_path"]
        model.write_bytes(model.read_bytes() + b"tamper")
    elif failure == "metadata_hash":
        metadata["dataset_sha256"] = "wrong"
    elif failure == "missing_model":
        metadata["models"].pop()
    elif failure == "features":
        metadata["models"][0]["features"] = ["target"]
    elif failure == "versions":
        metadata["versions"] = {}
    elif failure in {"seed", "threshold"}:
        metadata[failure] = 43 if failure == "seed" else .4
    elif failure == "split":
        metadata["models"][0]["split"]["partitions"]["train"]["rows"] = 31
    elif failure == "stored_metrics":
        metadata["models"][0]["validation_metrics"]["roc_auc"] = .9
    elif failure == "labels_hash":
        labels.source.write_bytes(labels.source.read_bytes() + b"\n")
    else:
        records = labels.frame.to_dict("records")
        records[0]["label_recorded_at"] = local_dataset.frame.iloc[0].prediction_at.isoformat()
        write_label_fixture(labels.source, local_dataset, records)
    write_json(path, metadata)
    def forbid(*args, **kwargs):
        raise AssertionError("No debe entrenar ante errores técnicos")
    monkeypatch.setattr(Pipeline, "fit", forbid)
    output = tmp_path / "blocked"
    with pytest.raises(Phase2Error):
        opt.optimize(local_dataset.source, None, labels.source, None, reference, output)
    assert not output.exists()


def test_configuration_budget_includes_original_and_rejects_adaptive_options(monkeypatch):
    opt.check_plan()
    assert {name: len(configs)+1 for name, configs in opt.CONFIGURATIONS.items()} == {"random_forest": 4, "xgboost": 6}
    monkeypatch.setattr(opt, "CONFIGURATIONS", {"random_forest": {str(i): {} for i in range(8)}, "xgboost": {"one": {}}})
    with pytest.raises(Phase2Error, match="Máximo"):
        opt.check_plan()
    monkeypatch.setattr(opt, "CONFIGURATIONS", {"random_forest": {"one": {"random_state": 43}}, "xgboost": {"one": {}}})
    with pytest.raises(Phase2Error, match="semilla"):
        opt.check_plan()


def test_months_use_booking_lima_and_do_not_use_appointment_month():
    frame = pd.DataFrame({"prediction_at": pd.to_datetime(["2026-05-01T04:30:00Z", "2026-05-02T05:00:00Z"], utc=True),
                          "appointment_at": pd.to_datetime(["2026-06-01T00:00:00Z"]*2, utc=True), "target": [0, 1]})
    months = opt.monthly_metrics(frame, [.2, .8])
    assert [row["month"] for row in months] == ["2026-04", "2026-05"]
    assert all(row["rows"] == 1 and row["roc_auc"] is None for row in months)
    assert all(row["date_basis"] == "prediction_at America/Lima" for row in months)


@pytest.mark.parametrize("ap_gain,auc_gain,brier_gain,monthly_gain,relevant", [
    (.005, .01, -.01, .005, False), (.02, -.003, -.01, .02, False),
    (.02, .01, .003, .02, False), (.02, .01, -.01, -.02, False), (.02, .01, -.01, .02, True),
])
def test_relevance_requires_material_gain_and_monthly_consistency(ap_gain, auc_gain, brier_gain, monthly_gain, relevant):
    original = {key: .5 for key in opt.METRICS}
    metrics = {**original, "pr_auc_average_precision": .5+ap_gain, "roc_auc": .5+auc_gain, "brier_score": .5+brier_gain}
    result = opt.compare_to_original(metrics, [{"month": "2026-06", "pr_auc_average_precision": .5+monthly_gain}],
                                     original, [{"month": "2026-06", "pr_auc_average_precision": .5}])
    assert (result["descriptive_relevance"] == "passes_predeclared_screen") == relevant
    assert result["selection_status"] == "no_definitive_model_selected"


def test_commercial_context_keeps_approximation_and_user_audit_separate():
    context = opt.COMMERCIAL_CONTEXT
    assert context["transition_months_approximate"] == ["2026-06", "2026-07"]
    assert context["exact_effective_date"] is None
    assert context["provided_audit_no_show_rates"]["2026-07"] == .0715
    assert "not recomputed" in context["source"]
    assert any("exclusive causality is not established" in message for message in context["limitations"])
    assert len(PREDICTOR_COLUMNS) == 19
