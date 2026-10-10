"""Smoke Uvicorn/curl en loopback; --extensions añade SHAP/lotes; apaga su proceso."""

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
from tempfile import TemporaryFile
from time import perf_counter, sleep

from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--extensions", action="store_true", help="Verify explain RF/XGB and both batch modes")
    args = parser.parse_args()
    if args.output.exists() or not 1024 <= args.port <= 65535:
        parser.error("Destino debe ser nuevo y puerto entre 1024 y 65535.")
    # Verify a free loopback port before launch; never attach to another service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    key = secrets.token_urlsafe(48)
    env = {**os.environ, "ML_API_ENV": "development", "ML_ALLOW_EXPERIMENTAL_INFERENCE": "true",
           "ML_API_MODE": "exploratory_demo", "ML_API_SERVICE_KEY": key,
           "ML_MODEL_REFERENCE_DIR": str(args.reference.resolve()), "ML_MODEL_MANIFEST_SHA256": args.manifest_sha256,
           "ML_EXPOSE_DOCS": "false", "ML_ENABLE_LOCAL_EXPLANATIONS": str(args.extensions).lower(),
           "ML_MAX_BATCH_ITEMS": "32", "ML_MAX_CONCURRENT_PREDICTIONS": "2"}
    # Not using --env-file: do not import database credentials from .env.
    env.pop("POSTGRES_PASSWORD", None)
    with TemporaryFile() as logs:
        started = perf_counter()
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "depi_ml.api.main:create_app", "--factory",
                                    "--host", "127.0.0.1", "--port", str(args.port), "--no-access-log",
                                    "--log-config", "examples/uvicorn_logging.json"], env=env, stdout=logs, stderr=logs)

        def curl(path, payload=None, authenticated=False):
            config = f'header = "X-Service-API-Key: {key}"\n' if authenticated else ""
            cmd = ["curl", "--silent", "--show-error", "--max-time", "10", "--config", "-",
                   "--write-out", "\n%{http_code}", f"http://127.0.0.1:{args.port}{path}"]
            if payload is not None:
                # All body data artificial. Secret transmitted through stdin, not argv.
                cmd += ["--header", "Content-Type: application/json", "--data", json.dumps(payload)]
            result = subprocess.run(cmd, input=config, text=True, capture_output=True)
            if result.returncode:
                raise ConnectionError("curl could not reach the demo")
            raw, code = result.stdout.rsplit("\n", 1)
            return {"status": int(code), "body": json.loads(raw)}

        try:
            deadline = perf_counter() + 30
            while True:
                if process.poll() is not None:
                    logs.seek(0)
                    # Startup logs contain hashes, but a bad inherited env could contain secrets.
                    print(logs.read().decode(errors="replace").replace(key, "[redacted]"), file=sys.stderr)
                    raise RuntimeError("Uvicorn could not start")
                try:
                    live = curl("/health/live")
                    break
                except ConnectionError:
                    if perf_counter() > deadline:
                        raise RuntimeError("Uvicorn startup timeout") from None
                    sleep(.1)
            report = {"scope": "real_loopback_uvicorn_curl_synthetic", "startup_seconds": perf_counter() - started,
                      "live": live, "ready": curl("/health/ready"),
                      "models": curl("/internal/v1/models", authenticated=True),
                      "unauthorized": curl("/internal/v1/models")}
            payload = {"feature_version": "BOOKING_BASELINE_V1", "feature_context": {"snapshot_type": "booking_creation"},
                       "features": synthetic_features()}
            for name, model in (("random_forest", "random_forest_full_v1"), ("xgboost", "xgboost_full_v1")):
                report[name] = curl("/internal/v1/predictions", {**payload, "model_key": model}, True)
            report["compare"] = curl("/internal/v1/predictions/compare", payload, True)
            if args.extensions:
                for name, model in (("rf", "random_forest_full_v1"), ("xgb", "xgboost_full_v1")):
                    report["explain_" + name] = curl("/internal/v1/predictions/explain",
                                                    {**payload, "model_key": model, "top_k": 5}, True)
                for mode in ("single", "compare"):
                    batch = {"mode": mode, "feature_version": payload["feature_version"],
                             "feature_context": payload["feature_context"], "items": [{"features": payload["features"]}] * 2}
                    report["batch_" + mode] = curl("/internal/v1/predictions/batch", batch, True)
                for operation in ("explain", "batch"):
                    report["unauthorized_" + operation] = curl("/internal/v1/predictions/" + operation, {})
                assert all(report[name]["status"] == 200 for name in ("explain_rf", "explain_xgb", "batch_single", "batch_compare"))
                assert all(report["unauthorized_" + name]["status"] == 401 for name in ("explain", "batch"))
            assert report["unauthorized"]["status"] == 401
            assert all(report[name]["status"] == 200 for name in ("live", "ready", "models", "random_forest", "xgboost", "compare"))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            print(json.dumps({name: item["status"] for name, item in report.items() if isinstance(item, dict)}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
