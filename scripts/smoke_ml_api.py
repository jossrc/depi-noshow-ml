"""Smoke Uvicorn/curl en loopback; --extensions añade SHAP/lotes; apaga su proceso."""

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
from tempfile import TemporaryFile
from time import perf_counter, sleep

from depi_ml.infrastructure.ml.synthetic_probe import synthetic_features
from depi_ml.datasets.schema import PREDICTOR_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--extensions", action="store_true", help="Verify explain RF/XGB and both batch modes")
    parser.add_argument("--research-risk",action="store_true",help="Verify published v2 policies and all synthetic bands")
    parser.add_argument("--policy",type=Path)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--expect-research-unavailable",action="store_true",help="Verify v2 flag off isolation on real HTTP")
    args = parser.parse_args()
    if args.output.exists() or not 1024 <= args.port <= 65535:
        parser.error("Destino debe ser nuevo y puerto entre 1024 y 65535.")
    if args.research_risk and (args.policy is None or args.policy_sha256 is None or args.expect_research_unavailable):
        parser.error("Research requiere --policy y pin explícito; incompatible con disabled check.")
    # Verify a free loopback port before launch; never attach to another service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    key = secrets.token_urlsafe(48)
    env = {**os.environ, "ML_API_ENV": "development", "ML_ALLOW_EXPERIMENTAL_INFERENCE": "true",
           "ML_API_MODE": "exploratory_demo", "ML_API_SERVICE_KEY": key,
           "ML_MODEL_REFERENCE_DIR": str(args.reference.resolve()), "ML_MODEL_MANIFEST_SHA256": args.manifest_sha256,
           "ML_EXPOSE_DOCS": "false", "ML_ENABLE_LOCAL_EXPLANATIONS": str(args.extensions).lower(),
           "ML_MAX_BATCH_ITEMS": "32", "ML_MAX_CONCURRENT_PREDICTIONS": "2",
           "ML_MAX_REQUEST_BYTES":"262144","ML_ENABLE_RESEARCH_RISK":str(args.research_risk).lower()}
    if args.research_risk:
        env.update(ML_RESEARCH_RISK_POLICY_PATH=str(args.policy.resolve()),ML_RESEARCH_RISK_POLICY_SHA256=args.policy_sha256)
    # Not using --env-file: do not import database credentials from .env.
    env.pop("POSTGRES_PASSWORD", None)
    with TemporaryFile() as logs:
        started = perf_counter()
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "depi_ml.api.main:create_app", "--factory",
                                    "--host", "127.0.0.1", "--port", str(args.port), "--no-access-log",
                                    "--log-config", "examples/uvicorn_logging.json"], env=env, stdout=logs, stderr=logs)

        def curl(path, payload=None, authenticated=False, *, raw_body=None,
                 invalid_key=False, duplicate_key=False):
            config = f'header = "X-Service-API-Key: {key}"\n' if authenticated else ""
            if invalid_key:
                config = 'header = "X-Service-API-Key: invalid-synthetic-key"\n'
            if duplicate_key:
                config += f'header = "X-Service-API-Key: {key}"\n'
            cmd = ["curl", "--silent", "--show-error", "--max-time", "10", "--config", "-",
                   "--write-out", "\n%{http_code}", f"http://127.0.0.1:{args.port}{path}"]
            if payload is not None or raw_body is not None:
                # All body data artificial. Secret transmitted through stdin, not argv.
                cmd += ["--header", "Content-Type: application/json"]
                serialized = raw_body if raw_body is not None else json.dumps(payload)
                # Also send the body through config stdin: large negative cases
                # must not exceed the operating system's per-argument limit.
                config += "data-binary = " + json.dumps(serialized) + "\n"
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
            for name in ("random_forest", "xgboost"):
                body = report[name]["body"]
                probability = body["probability_no_show"]
                assert math.isfinite(probability) and 0 <= probability <= 1
                assert body["risk_level"] is None and body["operational_use_allowed"] is False
                assert body["risk_policy_status"] == "pending_validation"
            assert [row["predictor_columns"] for row in report["models"]["body"]["models"]] == [PREDICTOR_COLUMNS] * 2
            comparison = report["compare"]["body"]
            for row, name in zip(comparison["results"], ("random_forest", "xgboost"), strict=True):
                assert row["model_key"] == report[name]["body"]["model_key"]
                assert math.isclose(row["probability_no_show"], report[name]["body"]["probability_no_show"], abs_tol=1e-10)
            assert comparison["comparison"]["ensemble_probability"] is None
            assert comparison["comparison"]["selected_winner"] is None
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
                negatives = {
                    "invalid_key": curl("/internal/v1/models", invalid_key=True),
                    "duplicate_key": curl("/internal/v1/models", authenticated=True, duplicate_key=True),
                    "invalid_json": curl("/internal/v1/predictions", authenticated=True, raw_body="{"),
                    "wrong_version": curl("/internal/v1/predictions", {**payload, "feature_version": "SYNTHETIC_INCOMPATIBLE"}, True),
                    "invalid_type": curl("/internal/v1/predictions", {**payload, "features": {**payload["features"], "clinic_id": "1"}}, True),
                    "extra_predictor": curl("/internal/v1/predictions", {**payload, "features": {**payload["features"], "previous_cancelled": 0}}, True),
                    "wrong_snapshot": curl("/internal/v1/predictions", {**payload, "feature_context": {"snapshot_type": "current"}}, True),
                    "invalid_top_k": curl("/internal/v1/predictions/explain", {**payload, "top_k": 11}, True),
                    "empty_batch": curl("/internal/v1/predictions/batch", {"mode": "single", "feature_version": payload["feature_version"], "feature_context": payload["feature_context"], "items": []}, True),
                    "oversized": curl("/internal/v1/predictions", authenticated=True, raw_body='"' + "x" * 262145 + '"'),
                }
                expected = {"invalid_key": 401, "duplicate_key": 401, "invalid_json": 400,
                            "wrong_version": 409, "invalid_type": 422, "extra_predictor": 422,
                            "wrong_snapshot": 422, "invalid_top_k": 422, "empty_batch": 422, "oversized": 413}
                assert all(negatives[name]["status"] == status for name, status in expected.items())
                report.update(negatives)
            if args.research_risk:
                report["research_catalog"]=curl("/internal/v2/risk-policy",authenticated=True)
                assert report["research_catalog"]["status"]==200 and report["research_catalog"]["body"]["available"]
                public_cases=json.loads(Path("examples/research_risk_synthetic_cases.json").read_text())
                report["research_cases"]=[]
                covered={"random_forest_full_v1":set(),"xgboost_full_v1":set()}
                for case in public_cases:
                    record={"name":case["name"],"synthetic_payload":case["payload"],"single":[]}
                    for model,expected_level in case["expected_levels"].items():
                        body={**case["payload"],"model_key":model}
                        legacy=curl("/internal/v1/predictions",body,True)
                        response=curl("/internal/v2/predictions",body,True)
                        assert legacy["status"]==response["status"]==200
                        result=response["body"]
                        assert math.isclose(result["probability_no_show"],legacy["body"]["probability_no_show"],rel_tol=0,abs_tol=1e-10)
                        assert result["experimental_risk"]["level"]==expected_level
                        assert result["experimental_risk"]["status"]=="RESEARCH_PROVISIONAL" and result["operational_use_allowed"] is False
                        covered[model].add(expected_level)
                        record["single"].append(response)
                    record["compare"]=curl("/internal/v2/predictions/compare",case["payload"],True)
                    assert record["compare"]["status"]==200
                    assert [r["experimental_risk"]["level"] for r in record["compare"]["body"]["results"]]==list(case["expected_levels"].values())
                    report["research_cases"].append(record)
                assert all(levels=={"LOW","MEDIUM","HIGH"} for levels in covered.values())
                report["research_coverage"]={k:sorted(v) for k,v in covered.items()}
                report["research_negatives"]={
                    "missing_key":curl("/internal/v2/risk-policy"),
                    "invalid_key":curl("/internal/v2/predictions",{},invalid_key=True),
                    "duplicate_key":curl("/internal/v2/risk-policy",authenticated=True,duplicate_key=True),
                    "invalid_json":curl("/internal/v2/predictions",authenticated=True,raw_body="{"),
                    "wrong_version":curl("/internal/v2/predictions",{**payload,"feature_version":"wrong"},True),
                    "extra_predictor":curl("/internal/v2/predictions",{**payload,"features":{**payload["features"],"previous_cancelled":0}},True),
                    "invalid_type":curl("/internal/v2/predictions",{**payload,"features":{**payload["features"],"clinic_id":"1"}},True),
                    "wrong_snapshot":curl("/internal/v2/predictions",{**payload,"feature_context":{"snapshot_type":"current"}},True),
                    "oversized":curl("/internal/v2/predictions",authenticated=True,raw_body='"'+"x"*262145+'"'),
                }
                expected=[401,401,401,400,409,422,422,422,413]
                assert [r["status"] for r in report["research_negatives"].values()]==expected
            elif args.expect_research_unavailable:
                report["research_disabled_catalog"]=curl("/internal/v2/risk-policy",authenticated=True)
                assert report["research_disabled_catalog"]["status"]==200
                assert report["research_disabled_catalog"]["body"]["status"]=="UNAVAILABLE"
                assert report["research_disabled_catalog"]["body"]["reason"]=="disabled"
                report["research_disabled_prediction"]=curl("/internal/v2/predictions",payload,True)
                assert report["research_disabled_prediction"]["status"]==503
                assert report["ready"]["status"]==200 and report["random_forest"]["status"]==200
            assert report["unauthorized"]["status"] == 401
            assert all(report[name]["status"] == 200 for name in ("live", "ready", "models", "random_forest", "xgboost", "compare"))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            print(json.dumps({name: item["status"] for name, item in report.items() if isinstance(item, dict) and "status" in item}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
