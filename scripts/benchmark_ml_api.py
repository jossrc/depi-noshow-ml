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
    args = parser.parse_args()
    if args.output.exists() or not 5 <= args.requests <= 100:
        parser.error("Destino debe ser nuevo y --requests entre 5 y 100.")
    settings = ApiSettings(allow_experimental_inference=True, service_key=secrets.token_urlsafe(48),
                          model_reference_dir=args.reference, model_manifest_sha256=args.manifest_sha256)
    payload = {"feature_version": "BOOKING_BASELINE_V1", "feature_context": {"snapshot_type": "booking_creation"},
               "features": synthetic_features()}
    headers = {"X-Service-API-Key": settings.service_key.get_secret_value(), "X-Request-ID": "synthetic-benchmark"}
    before, started = peak_rss_mib(), perf_counter()
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        report = {"scope": "local_synthetic_microbenchmark_not_SLA", "machine": platform.machine(),
                  "python": platform.python_version(), "startup_seconds": perf_counter() - started,
                  "peak_process_rss_before_mib": before, "peak_process_rss_after_load_mib": peak_rss_mib(),
                  "http_transport": "TestClient_in_process", "workers": 1, "max_concurrent_predictions": 2,
                  "models": [{"model_key": m.model_key, "sha256": m.model_sha256,
                              "status": m.validation_status} for m in client.app.state.registry.list_models()],
                  "http": {}, "pipeline": {}}

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
