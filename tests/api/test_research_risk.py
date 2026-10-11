"""Contrato v2 y compatibilidad v1; fakes explícitos para errores y admisión."""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from depi_ml.api.main import create_app
from depi_ml.api.openapi_versions import versioned_openapi
from depi_ml.domain.errors import RateLimited
from depi_ml.domain.research_risk_policy import ResearchRiskPolicy,canonical_hash
from depi_ml.infrastructure.ml.research_risk_registry import ResearchRiskPolicyRegistry
from depi_ml.infrastructure.config.api_settings import ApiSettings

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def research_client(settings,registry):
    raw=json.loads((ROOT/"policies/research_risk_v1_1/risk_policy.json").read_text())["policies"]
    policies=[]
    for p in raw:
        info=registry.get(p["model_key"]).info
        p.update(model_sha256=info.model_sha256,model_version=info.model_version)
        p["source"].update(dataset_sha256=info.dataset_sha256,experiment_manifest_sha256=info.manifest_sha256)
        p["policy_hash_sha256"]=canonical_hash({k:v for k,v in p.items() if k!="policy_hash_sha256"})
        policies.append(ResearchRiskPolicy.model_validate_json(json.dumps(p)))
    policy_registry=ResearchRiskPolicyRegistry(policies,registry)
    with TestClient(create_app(settings.model_copy(update={"enable_research_risk":True}),
                             registry_loader=lambda _:registry,risk_policy_loader=lambda *_:policy_registry)) as c:
        yield c


def test_single_compare_catalog_keep_raw_v1_values(research_client,headers,payload):
    c=research_client
    catalog=c.get("/internal/v2/risk-policy",headers=headers).json()
    assert catalog["available"] and len(catalog["policies"])==2
    for key in ["random_forest_full_v1","xgboost_full_v1"]:
        body={**payload,"model_key":key}
        old=c.post("/internal/v1/predictions",json=body,headers=headers).json()
        new=c.post("/internal/v2/predictions",json=body,headers=headers).json()
        assert new["probability_no_show"]==old["probability_no_show"]
        assert new["probability_no_show_percent"]==new["probability_no_show"]*100
        assert new["risk_level"] is None and new["risk_policy_status"]=="pending_validation"
        assert new["experimental_risk"]["status"]=="RESEARCH_PROVISIONAL"
        assert new["experimental_risk"]["operational_use_allowed"] is False
        assert new["api_version"]=="v2" and new["request_id"]==headers["X-Request-ID"]
    r=c.post("/internal/v2/predictions/compare",json=payload,headers=headers)
    assert r.status_code==200
    rows=r.json()["results"]
    assert [row["experimental_risk"]["level"] for row in rows]==["MEDIUM","MEDIUM"]
    assert rows[0]["experimental_risk"]["policy_hash"]!=rows[1]["experimental_risk"]["policy_hash"]
    assert r.json()["comparison"]["selected_winner"] is None


@pytest.mark.parametrize("path",["/internal/v2/risk-policy","/internal/v2/predictions","/internal/v2/predictions/compare"])
@pytest.mark.parametrize("mode",["missing","wrong","duplicate"])
def test_v2_auth_before_body(research_client,headers,path,mode):
    h=[] if mode=="missing" else [("X-Service-API-Key","wrong")] if mode=="wrong" else [("X-Service-API-Key",headers["X-Service-API-Key"])]*2
    r=research_client.get(path,headers=h) if path.endswith("risk-policy") else research_client.post(path,headers=h,content="{")
    assert r.status_code==401


@pytest.mark.parametrize("kind,expected",[("json",400),("version",409),("extra",422),("type",422),("snapshot",422),("oversized",413),("rate",429)])
def test_v2_errors_sanitized(research_client,headers,payload,registry,kind,expected):
    body=copy.deepcopy(payload)
    if kind=="version":body["feature_version"]="wrong"
    if kind=="extra":body["features"]["private-extra-field"]="private-value"
    if kind=="type":body["features"]["clinic_id"]="private-value"
    if kind=="snapshot":body["feature_context"]["snapshot_type"]="current"
    if kind=="rate":registry.get("random_forest_full_v1").probability=RateLimited()
    kwargs={"json":body} if kind not in ["json","oversized"] else {"content":"{" if kind=="json" else "x"*262145}
    r=research_client.post("/internal/v2/predictions",headers={**headers,"Content-Type":"application/json"},**kwargs)
    assert r.status_code==expected
    assert all(v not in r.text for v in ["private-value","private-extra-field",headers["X-Service-API-Key"],"Traceback"])


@pytest.mark.parametrize("env,flag",[("development",False),("production",True),("staging",True)])
def test_disabled_isolated_from_legacy(settings,registry,headers,payload,env,flag):
    def forbidden(*_):
        pytest.fail("Disabled policy must not be loaded")
    config=settings.model_copy(update={"env":env,"enable_research_risk":flag})
    with TestClient(create_app(config,registry_loader=lambda _:registry,risk_policy_loader=forbidden)) as c:
        catalog=c.get("/internal/v2/risk-policy",headers=headers).json()
        assert catalog["status"]=="UNAVAILABLE" and catalog["policies"]==[]
        assert c.post("/internal/v2/predictions",json=payload,headers=headers).status_code==503
        assert c.get("/health/ready").status_code==(200 if env=="development" else 503)
        assert c.post("/internal/v1/predictions",json=payload,headers=headers).status_code==(200 if env=="development" else 503)


@pytest.mark.parametrize("failure",["missing_policy","hash_mismatch","invalid_policy","model_identity_mismatch","insufficient_support"])
def test_policy_failure_never_breaks_v1(settings,registry,headers,payload,failure,caplog):
    from depi_ml.domain.research_risk_policy import RiskPolicyUnavailable
    def fail(*_):
        raise RiskPolicyUnavailable(failure)
    with TestClient(create_app(settings.model_copy(update={"enable_research_risk":True}),
                             registry_loader=lambda _:registry,risk_policy_loader=fail)) as c:
        assert c.get("/health/ready").status_code==200
        assert c.post("/internal/v1/predictions",json=payload,headers=headers).status_code==200
        r=c.post("/internal/v2/predictions/compare",json=payload,headers=headers)
        assert r.status_code==503 and r.json()["error"]["code"]=="RESEARCH_RISK_POLICY_NOT_READY"
        assert c.get("/internal/v2/risk-policy",headers=headers).json()["reason"]==failure
        assert headers["X-Service-API-Key"] not in caplog.text+r.text


def test_versioned_openapi_and_strict_flag(research_client,headers,monkeypatch):
    old=json.loads((ROOT/"docs/contracts/ml-internal-v1.openapi.json").read_text())
    new=json.loads((ROOT/"docs/contracts/ml-internal-v2.openapi.json").read_text())
    assert research_client.app.openapi()==old
    assert versioned_openapi(research_client.app,"v2")==new
    assert research_client.get("/internal/v2/openapi.json",headers=headers).json()==new
    assert set(new["paths"])=={"/internal/v2/risk-policy","/internal/v2/predictions","/internal/v2/predictions/compare"}
    assert len(new["components"]["schemas"]["PredictionFeatures"]["required"])==19
    for value in ["1","TRUE","yes","", "secret-not-echoed"]:
        monkeypatch.setenv("ML_ENABLE_RESEARCH_RISK",value)
        with pytest.raises(ValueError) as e:
            ApiSettings.from_env()
        if value:
            assert value not in str(e.value)


@pytest.mark.parametrize("kind,reason",[("malformed_pin","invalid_policy"),("wrong_pin","hash_mismatch"),
                                        ("missing","missing_policy"),("invalid_json","invalid_policy")])
def test_actual_optional_loader_config_failures_preserve_v1(settings,registry,headers,payload,tmp_path,kind,reason):
    import hashlib
    path=ROOT/"policies/research_risk_v1_1/risk_policy.json"
    pin="invalid" if kind=="malformed_pin" else "0"*64
    if kind=="missing":path=tmp_path/"absent.json"
    if kind=="invalid_json":
        path=tmp_path/"invalid.json";path.write_text("{");pin=hashlib.sha256(path.read_bytes()).hexdigest()
    config=ApiSettings(**{**settings.model_dump(),"enable_research_risk":True,
                          "research_risk_policy_path":path,"research_risk_policy_sha256":pin})
    with TestClient(create_app(config,registry_loader=lambda _:registry)) as c:
        assert c.get("/health/ready").status_code==200
        assert c.post("/internal/v1/predictions",json=payload,headers=headers).status_code==200
        assert c.get("/internal/v2/risk-policy",headers=headers).json()["reason"]==reason
        assert c.post("/internal/v2/predictions",json=payload,headers=headers).status_code==503
