"""Estudio offline reproducible; solo validación original, sin test ni SQL."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from depi_ml.analysis.dataset import digest, load_dataset, write_json
from depi_ml.datasets.label_availability import attach_labels, load_label_availability
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.research_risk_policy import LEVELS,band_index,canonical_hash,POLICY_VERSION
from depi_ml.evaluation.metrics import binary_metrics
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerifier
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.sklearn_predictor import positive_class_index
from depi_ml.training.splits import temporal_split

SEED, REPLICATES = 42, 1000
JUNE = pd.Timestamp("2026-06-01T00:00:00-05:00")
WARNINGS = (
    "Relative raw-score bands, not validated absolute risks or cancellation probabilities.",
    "Original model/hyperparameters already exposed to April-June validation; June is not independent model evaluation.",
    "Historical booking/catalog snapshots and actual label availability remain unaccredited.",
    "Approximate June-July business transition; post-change performance is unknown without independent evaluation.",
    "No operational approval; no automatic commercial actions. Calibration candidates are offline only.",
)


def normalized(value):
    # Quantization for reproducible artifacts, not for served probabilities.
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Nonfinite research artifact")
        return round(value, 12)
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def thresholds(scores):
    p = np.asarray(scores)
    if p.ndim != 1 or p.dtype.kind not in "fi" or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid quantile scores")
    low, high = map(float, np.round(np.quantile(p, [1/3, 2/3], method="linear"), 12))
    if not 0 < low < high < 1:
        raise ValueError("Degenerate research thresholds")
    return low, high


def assignments(scores, low, high):
    return np.fromiter((band_index(float(p),low,high) for p in scores),dtype=int,count=len(scores))


def calibration(y, p):
    cells = []
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        if mask.any():
            cells.append({"lower": i/10, "upper": (i+1)/10, "rows": int(mask.sum()),
                          "positives": int(y[mask].sum()), "mean_probability": float(p[mask].mean()),
                          "observed_rate": float(y[mask].mean())})
    return {"bins": "10 fixed uniform; last includes 1", "cells": cells,
            "mean_probability": float(p.mean()), "prevalence": float(y.mean()),
            "ece_uniform_10": sum(c["rows"]/len(y)*abs(c["mean_probability"]-c["observed_rate"]) for c in cells),
            "brier_score": float(np.mean((p-y)**2)), "log_loss": float(log_loss(y, p, labels=[0,1]))}


def band_metrics(frame, p, low, high, *, model_key, period, clinic_id=None):
    y = frame.target.to_numpy(dtype=int)
    band = assignments(p, low, high)
    clients, inverse = np.unique(frame.client_id.to_numpy(), return_inverse=True)
    counts = np.bincount(inverse*3+band, minlength=len(clients)*3).reshape(-1,3)
    positives = np.bincount(inverse*3+band, weights=y, minlength=len(clients)*3).reshape(-1,3)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(REPLICATES):
        sampled = rng.integers(0, len(clients), size=len(clients))
        denominator = counts[sampled].sum(axis=0)
        estimates.append(np.divide(positives[sampled].sum(axis=0), denominator,
                                  out=np.full(3,np.nan), where=denominator>0))
    estimates = np.asarray(estimates)
    rows = []
    for index, level in enumerate(LEVELS):
        mask = band==index
        n, pos = int(mask.sum()), int(y[mask].sum())
        n_clients = int((counts[:,index]>0).sum())
        if clinic_id is not None and n < 30:
            continue  # Suppress small site cells, including their counts/rates.
        valid = estimates[:,index][np.isfinite(estimates[:,index])]
        ci = np.quantile(valid,[.025,.975]).tolist() if n_clients>=20 and len(valid)>=950 else None
        rows.append({"model_key":model_key,"period":period,"clinic_id":clinic_id,"level":level,
                     "rows":n,"positives":pos,"observed_rate":pos/n if n else None,
                     "coverage":n/len(y),"clients":n_clients,"ci95":ci,
                     "ci_method":"client_cluster_percentile_1000_seed42_conditional_on_fixed_model_and_thresholds",
                     "precision_if_band_selected":pos/n if n else None,
                     "recall_of_all_positive_events":pos/int(y.sum()) if y.sum() else None,
                     "false_positives_if_band_selected":n-pos,"positive_events_outside_band":int(y.sum())-pos})
    return rows


def build(reference, manifest_sha256, csv_path, labels_path, output):
    output = Path(output).resolve()
    if not output.is_relative_to(Path.cwd().resolve()/"reports") or output.exists():
        raise ValueError("New output directory under reports required")
    ArtifactVerifier(reference,manifest_sha256).verify()
    metadata = json.loads((reference/"experiment.json").read_text())
    if digest(reference/"experiment.json") != manifest_sha256:
        raise ValueError("Trusted manifest changed")
    dataset = load_dataset(csv_path)
    labels = load_label_availability(dataset,labels_path)
    if dataset.sha256 != metadata["dataset_sha256"] or labels.identity()!=metadata["label_availability"]:
        raise ValueError("Original source identity mismatch")
    recorded = next(e["split"] for e in metadata["models"] if e["experiment"]=="full" and e["model"]=="random_forest")
    if any(e["split"]!=recorded for e in metadata["models"] if e["experiment"]=="full"):
        raise ValueError("Original models have different temporal partitions")
    if (recorded["validation_start"] != "2026-04-01T05:00:00+00:00"
            or recorded["test_start"] != "2026-07-01T05:00:00+00:00"):
        raise ValueError("Unsupported original cohort boundaries")
    split = temporal_split(attach_labels(dataset,labels),recorded["validation_start"],recorded["test_start"],
                           recorded["test_end"],dataset.manifest["exported_at"],include_test=False)
    if split.test is not None or split.metadata!=recorded:
        raise ValueError("Original split mismatch")
    frame = split.validation
    development = frame.prediction_at < JUNE
    diagnostic = ~development
    if any(int(mask.sum())<1000 or frame.loc[mask,"target"].nunique()!=2 for mask in [development,diagnostic]):
        raise ValueError("Insufficient cohort support")
    registry = ModelRegistry.load(ApiSettings(model_reference_dir=reference,model_manifest_sha256=manifest_sha256))
    bands, policies, model_comparison, diagnostics = [], [], [], []
    for info in registry.list_models():
        pipeline = registry.get(info.model_key).pipeline
        p = pipeline.predict_proba(frame[PREDICTOR_COLUMNS])[:,positive_class_index(pipeline)]
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError("Invalid original probabilities")
        y = frame.target.to_numpy(dtype=int)
        metrics = binary_metrics(y,p,.5)
        original = next(e["validation_metrics"] for e in metadata["models"] if e["experiment"]=="full" and e["model"]==info.algorithm)
        for name in ["roc_auc","pr_auc_average_precision","precision","recall","f1","brier_score"]:
            if not np.isclose(metrics[name],original[name],atol=1e-12,rtol=1e-12):
                raise ValueError("Original metrics not reproduced")
        if metrics["confusion_matrix"]!=original["confusion_matrix"]:
            raise ValueError("Original confusion matrix not reproduced")
        low,high = thresholds(p[development])
        overall = []
        monthly = frame.prediction_at.dt.tz_convert("America/Lima").dt.strftime("%Y-%m").to_numpy()
        for period,mask in [("development_april_may",development.to_numpy()),("diagnostic_june",diagnostic.to_numpy()),
                            *[(month,monthly==month) for month in sorted(set(monthly))]]:
            cells = band_metrics(frame.loc[mask],p[mask],low,high,model_key=info.model_key,period=period)
            bands.extend(cells)
            if period in ["development_april_may","diagnostic_june"]:
                overall.extend(cells)
            for clinic in sorted(frame.loc[mask,"clinic_id"].unique()):
                site_mask = mask & (frame.clinic_id.to_numpy()==clinic)
                if int(site_mask.sum())>=90:
                    bands.extend(band_metrics(frame.loc[site_mask],p[site_mask],low,high,
                                              model_key=info.model_key,period=period,clinic_id=int(clinic)))
        support = all(r["rows"]>=100 and r["clients"]>=20 and r["ci95"] is not None for r in overall)
        for period in ["development_april_may","diagnostic_june"]:
            rates = [r["observed_rate"] for r in overall if r["period"]==period]
            support = support and len(rates)==3 and rates[0]<rates[1]<rates[2]
        raw_diagnostics = {"model_key":info.model_key,"validation":calibration(y,p),
                           "development":calibration(y[development],p[development]),
                           "june":calibration(y[diagnostic],p[diagnostic])}
        # Calibration fit labels must be known before June starts, unlike threshold quantiles.
        cal_fit = development.to_numpy() & (frame.label_recorded_at<JUNE).to_numpy()
        if int(cal_fit.sum())<1000 or np.unique(y[cal_fit]).size!=2:
            raise ValueError("Insufficient temporally mature calibration support")
        logit = np.log(np.clip(p.astype(float),1e-12,1-1e-12)/(1-np.clip(p.astype(float),1e-12,1-1e-12)))
        sigmoid = LogisticRegression(C=1e6,max_iter=1000,solver="lbfgs",random_state=SEED).fit(logit[cal_fit,None],y[cal_fit])
        isotonic = IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(p[cal_fit],y[cal_fit])
        raw_diagnostics["offline_calibration_candidates"] = {
            "status":"EXPLORATORY_ONLY_NOT_SERVED","fit_rows":int(cal_fit.sum()),
            "fit_labels_before":JUNE.isoformat(),"evaluation_rows":int(diagnostic.sum()),
            "sigmoid_logit_C_1e6":calibration(y[diagnostic],sigmoid.predict_proba(logit[diagnostic,None])[:,1]),
            "isotonic":calibration(y[diagnostic],isotonic.predict(p[diagnostic])),
            "limitations":list(WARNINGS),"original_pipeline_refitted":False}
        diagnostics.append(raw_diagnostics)
        model_comparison.append({"model_key":info.model_key,"cohort":"original April-June validation",**metrics,
                                 "log_loss":raw_diagnostics["validation"]["log_loss"]})
        policy = normalized({"policy_id":info.model_key+"_tertiles_202604_202605",
            "policy_version":POLICY_VERSION,"model_key":info.model_key,"model_version":info.model_version,
            "model_sha256":info.model_sha256,"feature_version":"BOOKING_BASELINE_V1",
            "prediction_target":"NO_SHOW_CANDIDATE","target_class":1,"score_space":"raw_probability_no_show",
            "low_upper_threshold":low,"high_lower_threshold":high,
            "boundary_rule":"LOW:p<t_low;MEDIUM:t_low<=p<t_high;HIGH:p>=t_high",
            "method":"score_quantiles_tertiles_exploratory","quantile_interpolation":"linear",
            "threshold_decimal_places":12,"seed":SEED,
            "development_start":"2026-04-01T00:00:00-05:00","development_end_exclusive":JUNE.isoformat(),
            "diagnostic_end_exclusive":"2026-07-01T00:00:00-05:00",
            "label_cutoff_exclusive":recorded["test_start"],
            "source":{"dataset_sha256":dataset.sha256,"dataset_manifest_sha256":dataset.manifest_sha256,
                      "labels_sha256":labels.sha256,"labels_manifest_sha256":labels.manifest_sha256,
                      "experiment_manifest_sha256":manifest_sha256},
            "development_rows":int(development.sum()),"development_positives":int(y[development].sum()),
            "development_prevalence":float(y[development].mean()),"band_evidence":overall,
            "calibration_ece":raw_diagnostics["validation"]["ece_uniform_10"],
            "support_passed":bool(support),"status":"RESEARCH_PROVISIONAL" if support else "RESEARCH_CANDIDATE_NOT_SUPPORTED",
            "meaning":"relative_score_band_not_validated_absolute_risk",
            "operational_use_allowed":False,"approved_by":None,"warnings":list(WARNINGS)})
        policy["policy_hash_sha256"] = canonical_hash(policy)
        policies.append(policy)
    protected = {dataset.source:dataset.sha256,dataset.manifest_source:dataset.manifest_sha256,
                 labels.source:labels.sha256,labels_path.with_name(labels_path.stem+"_manifest.json"):labels.manifest_sha256,
                 reference/"experiment.json":manifest_sha256}
    if any(digest(path)!=sha for path,sha in protected.items()):
        raise ValueError("Original source changed during research")
    output.mkdir(parents=True,exist_ok=False)
    bundle = normalized({"artifact_type":"research_risk_policy","format_version":1,"policies":policies})
    write_json(output/"risk_policy.json",bundle)
    write_json(output/"risk_bands_metrics.json",normalized(bands))
    write_json(output/"calibration_diagnostics.json",normalized(diagnostics))
    write_json(output/"model_comparison.json",normalized(model_comparison))
    with (output/"risk_bands_metrics.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(bands[0]));writer.writeheader();writer.writerows(normalized(bands))
    write_json(output/"build_verification.json",{"test_materialized":False,"test_predictions":0,
        "original_models_refitted":False,"offline_calibrators_fit":4,"sources_unchanged":True,
        "risk_bands_supported":all(p["support_passed"] for p in policies),
        "policy_file_sha256":digest(output/"risk_policy.json"),"versions":metadata["versions"],
        "builder_sha256":digest(Path(__file__)),"cluster_bootstrap_replicates":REPLICATES,
        "minimum_site_cell_rows":30,"minimum_cluster_ci_clients":20})
    return bundle
