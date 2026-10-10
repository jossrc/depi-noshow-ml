"""Microbenchmark opt-in sobre una fixture artificial; no evalúa datasets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import resource
import secrets
import statistics
import sys
from time import perf_counter

from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features


def peak_rss_mib():
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def summary(milliseconds):
    import numpy as np
    return {"n": len(milliseconds), "mean_ms": statistics.mean(milliseconds),
            "p50_ms": float(np.percentile(milliseconds, 50)), "p95_ms": float(np.percentile(milliseconds, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--extensions", action="store_true", help="Measure real explain and batches 1/8/32")
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.requests <= 100:
        parser.error("Destino debe ser nuevo y --requests entre 5 y 100.")
    settings = ApiSettings(allow_experimental_inference=True, service_key=secrets.token_urlsafe(48),
                          model_reference_dir=args.reference, model_manifest_sha256=args.manifest_sha256,
                          enable_local_explanations=args.extensions)
    payload = {"feature_version": "BOOKING_BASELINE_V1", "feature_context": {"snapshot_type": "booking_creation"},
               "features": synthetic_features()}
    headers = {"X-Service-API-Key": settings.service_key.get_secret_value(), "X-Request-ID": "synthetic-benchmark"}
    before, started = peak_rss_mib(), perf_counter()
    initialization = {}
    def load_explainers(registry):
        from depi_ml.infrastructure.ml.shap_explainer import load_explainers as load
        initialization["peak_rss_models_only_mib"] = peak_rss_mib()
        start = perf_counter()
        result = load(registry)
        assert set(result) == set(MODEL_KEYS)
        initialization["explainer_initialization_seconds"] = perf_counter() - start
        initialization["peak_rss_with_explainers_mib"] = peak_rss_mib()
        return result
    with TestClient(create_app(settings, explainer_loader=load_explainers)) as client:
        assert client.get("/health/ready").status_code == 200
        report = {"scope": "local_synthetic_microbenchmark_not_SLA", "machine": platform.machine(),
                  "python": platform.python_version(), "startup_seconds": perf_counter() - started,
                  "peak_process_rss_before_mib": before, "peak_process_rss_after_load_mib": peak_rss_mib(),
                  "http_transport": "TestClient_in_process", "workers": 1, "max_concurrent_predictions": 2,
                  "models": [{"model_key": m.model_key, "sha256": m.model_sha256,
                              "status": m.validation_status} for m in client.app.state.registry.list_models()],
                  "http": {}, "pipeline": {}}
        report.update(initialization)
        report["platform"] = platform.platform()
        import importlib.metadata
        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("fastapi", "pydantic", "starlette", "httpx", "numpy", "pandas", "scikit-learn", "xgboost", "joblib")}
        if args.extensions:
            report["versions"]["shap"] = importlib.metadata.version("shap")

        def timed_request(key=None, compare=False):
            body = payload if compare else {**payload, "model_key": key}
            start = perf_counter()
            response = client.post("/internal/v1/predictions" + ("/compare" if compare else ""), json=body, headers=headers)
            assert response.status_code == 200
            return (perf_counter() - start) * 1000, response.json()

        for key in MODEL_KEYS:
            for _ in range(3):
                timed_request(key)
            report["http"][key] = summary([timed_request(key)[0] for _ in range(args.requests)])
        report["http"]["compare"] = summary([timed_request(compare=True)[0] for _ in range(args.requests)])
        report["synthetic_compare_response"] = timed_request(compare=True)[1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            report["http"]["compare_concurrency_2"] = summary(list(pool.map(
                lambda _: timed_request(compare=True)[0], range(args.requests))))
        if args.extensions:
            from depi_ml.api.schemas.examples import explanation_examples
            def timed_extension(suffix, body):
                start = perf_counter()
                response = client.post("/internal/v1/predictions/" + suffix, json=body, headers=headers)
                assert response.status_code == 200, response.text
                return (perf_counter() - start) * 1000, response.json()
            for name, example in explanation_examples().items():
                timed_extension("explain", example["value"])
                report["http"]["explain_" + name] = summary([
                    timed_extension("explain", example["value"])[0] for _ in range(args.requests)])
                report["synthetic_explanation_" + name] = timed_extension("explain", example["value"])[1]
            for mode in ("single", "compare"):
                for size in (1, 8, 32):
                    body = {"mode": mode, "feature_version": payload["feature_version"],
                            "feature_context": payload["feature_context"], "items": [{"features": payload["features"]}] * size}
                    timed_extension("batch", body)
                    report["http"][f"batch_{mode}_{size}"] = summary([
                        timed_extension("batch", body)[0] for _ in range(args.requests)])
            # An explain and an independent XGB prediction share the global two
            # slots; concurrent explains are intentionally rejected with 429.
            def mixed_pair():
                with ThreadPoolExecutor(max_workers=2) as pair:
                    a = pair.submit(timed_extension, "explain", explanation_examples()["rf"]["value"])
                    b = pair.submit(timed_request, MODEL_KEYS[1])
                    return a.result()[0], b.result()[0]
            mixed = [mixed_pair() for _ in range(args.requests)]
            report["http"]["explain_rf_with_xgb_concurrency_2"] = summary([a for a, _ in mixed])
            report["http"]["xgb_with_explain_rf_concurrency_2"] = summary([b for _, b in mixed])
            report["extension_warmup_per_case"] = 1
            report["max_concurrent_explanations"] = 1
            report["batch_execution"] = "sequential_rows_one_global_admission_slot"
        import pandas as pd
        frame = pd.DataFrame([payload["features"]], columns=PREDICTOR_COLUMNS)
        for key in MODEL_KEYS:
            pipeline = client.app.state.registry.get(key).pipeline
            times = []
            for _ in range(args.requests):
                start = perf_counter()
                pipeline.predict_proba(frame)
                times.append((perf_counter() - start) * 1000)
            report["pipeline"][key] = summary(times)
        report["peak_process_rss_after_requests_mib"] = peak_rss_mib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("scope", "startup_seconds", "peak_process_rss_after_load_mib", "http", "pipeline")}, indent=2))


if __name__ == "__main__":
    main()
