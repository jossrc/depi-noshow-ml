"""Microbenchmark sintético v2 en proceso; no SLA ni throughput productivo."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import secrets
import sys
from time import perf_counter

import numpy as np
from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.infrastructure.config.api_settings import ApiSettings


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["reference","policy","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ["manifest-sha256","policy-sha256"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--requests",type=int,default=20)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path.cwd().resolve()/"reports") or not 5<=args.requests<=100:
        parser.error("Destino nuevo en reports y 5–100 requests requeridos.")
    settings=ApiSettings(allow_experimental_inference=True,service_key=secrets.token_urlsafe(48),
        enable_research_risk=True,model_reference_dir=args.reference,model_manifest_sha256=args.manifest_sha256,
        research_risk_policy_path=args.policy,research_risk_policy_sha256=args.policy_sha256)
    payload=json.loads(Path("examples/prediction_compare.json").read_text())
    headers={"X-Service-API-Key":settings.service_key.get_secret_value()}
    report={"scope":"synthetic_TestClient_not_SLA","workers":1,"concurrency":2,"n":args.requests,"http":{},"rss_before_mib":rss()}
    start=perf_counter()
    with TestClient(create_app(settings)) as client:
        report.update(startup_seconds=perf_counter()-start,rss_loaded_mib=rss())
        assert client.get("/health/ready").status_code==200
        def request(name):
            start=perf_counter()
            if name=="catalog":
                response=client.get("/internal/v2/risk-policy",headers=headers)
            elif name=="compare":
                response=client.post("/internal/v2/predictions/compare",json=payload,headers=headers)
            else:
                response=client.post("/internal/v2/predictions",json={**payload,"model_key":name},headers=headers)
            assert response.status_code==200
            return (perf_counter()-start)*1000
        def summary(values):
            return {"n":len(values),"p50_ms":float(np.percentile(values,50)),"p95_ms":float(np.percentile(values,95))}
        for name in ["catalog","random_forest_full_v1","xgboost_full_v1","compare"]:
            for _ in range(3):request(name)
            report["http"][name]=summary([request(name) for _ in range(args.requests)])
        with ThreadPoolExecutor(max_workers=2) as pool:
            report["http"]["compare_concurrency_2"]=summary(list(pool.map(lambda _:request("compare"),range(args.requests))))
        report["rss_after_requests_mib"]=rss()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
