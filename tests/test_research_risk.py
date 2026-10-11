"""Políticas públicas agregadas; sin datos privados, SQL ni reentrenamiento real."""

import copy
from dataclasses import replace, FrozenInstanceError
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from depi_ml.domain.model_info import ModelInfo
from depi_ml.domain.research_risk_policy import ResearchRiskPolicy, ResearchRiskClassifier, RiskPolicyUnavailable, canonical_hash
from depi_ml.datasets.schema import PREDICTOR_COLUMNS
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.research_risk_registry import ResearchRiskPolicyRegistry
from depi_ml.infrastructure.config.api_settings import ApiSettings

POLICY_PATH=Path(__file__).resolve().parents[1]/"policies/research_risk_v1_1/risk_policy.json"


@pytest.fixture
def raw_policies():
    return json.loads(POLICY_PATH.read_text())["policies"]


def parsed(raw):
    return ResearchRiskPolicy.model_validate_json(json.dumps(raw))


def info_for(policy):
    return ModelInfo(policy.model_key,"random_forest" if policy.model_key.startswith("random") else "xgboost",
                     policy.model_sha256,policy.source.experiment_manifest_sha256,"0"*64,
                     policy.source.dataset_sha256,tuple(PREDICTOR_COLUMNS))


def rehash(raw):
    raw["policy_hash_sha256"]=canonical_hash({k:v for k,v in raw.items() if k!="policy_hash_sha256"})
    return raw


@pytest.mark.parametrize("model",[0,1])
def test_all_exact_and_neighbor_boundaries(raw_policies,model):
    p=parsed(raw_policies[model]);info=info_for(p)
    cases=[(0.,"LOW"),(1.,"HIGH"),
           (math.nextafter(p.low_upper_threshold,-math.inf),"LOW"),(p.low_upper_threshold,"MEDIUM"),
           (math.nextafter(p.low_upper_threshold,math.inf),"MEDIUM"),
           (math.nextafter(p.high_lower_threshold,-math.inf),"MEDIUM"),(p.high_lower_threshold,"HIGH"),
           (math.nextafter(p.high_lower_threshold,math.inf),"HIGH")]
    assert [ResearchRiskClassifier.classify(v,p,info) for v,_ in cases]==[expected for _,expected in cases]


@pytest.mark.parametrize("value",[float("nan"),float("inf"),-float("inf"),-.1,1.1,True,False,"0.4",None,{},[]])
def test_invalid_probability_rejected(raw_policies,value):
    p=parsed(raw_policies[0])
    with pytest.raises(ValueError):
        ResearchRiskClassifier.classify(value,p,info_for(p))


def test_models_use_their_own_thresholds(raw_policies):
    rf,xgb=map(parsed,raw_policies)
    probability=(rf.low_upper_threshold+xgb.low_upper_threshold)/2
    assert ResearchRiskClassifier.classify(probability,rf,info_for(rf))=="LOW"
    assert ResearchRiskClassifier.classify(probability,xgb,info_for(xgb))=="MEDIUM"
    with pytest.raises(RiskPolicyUnavailable):
        ResearchRiskClassifier.classify(probability,rf,info_for(xgb))


@pytest.mark.parametrize("field,value",[("low_upper_threshold",0.),("low_upper_threshold",.8),
    ("high_lower_threshold",1.),("high_lower_threshold",-.1),("low_upper_threshold",float("nan")),
    ("feature_version","wrong"),("model_sha256","invalid"),("policy_hash_sha256","0"*64),
    ("support_passed",False),("status","APPROVED"),("approved_by","invented"),
    ("operational_use_allowed",True),("development_start","2026-04-01"),("development_rows",0)])
def test_invalid_or_changed_policy_rejected(raw_policies,field,value):
    raw=copy.deepcopy(raw_policies[0]);raw[field]=value
    with pytest.raises((ValidationError,ValueError)):
        parsed(raw)


def test_equal_thresholds_rejected_even_with_new_hash(raw_policies):
    raw=raw_policies[0];raw["high_lower_threshold"]=raw["low_upper_threshold"]
    with pytest.raises(ValidationError):
        parsed(rehash(raw))


def test_evidence_and_hash_not_just_self_signed(raw_policies):
    raw=raw_policies[0];raw["band_evidence"][0]["positives"]+=1
    with pytest.raises(ValidationError):
        parsed(rehash(raw))
    raw=raw_policies[1];raw["band_evidence"][0]["level"]="HIGH"
    with pytest.raises(ValidationError):
        parsed(rehash(raw))


def test_canonical_hash_and_immutability(raw_policies):
    p=parsed(raw_policies[0]);raw=p.model_dump(mode="json",exclude={"policy_hash_sha256"})
    assert canonical_hash(raw)==canonical_hash(dict(reversed(list(raw.items()))))==p.policy_hash_sha256
    with pytest.raises(ValidationError):
        p.low_upper_threshold=.1
    with pytest.raises(ValidationError):
        p.source.dataset_sha256="1"*64
    with pytest.raises(TypeError):
        p.band_evidence[0]=p.band_evidence[1]


def test_registry_rejects_cross_model_and_source_identity(raw_policies):
    policies=tuple(map(parsed,raw_policies));info=tuple(info_for(p) for p in policies)
    registry=ModelRegistry(SimpleNamespace(info=i) for i in info)
    r=ResearchRiskPolicyRegistry(policies,registry)
    assert r.list_policies()==policies
    with pytest.raises(FrozenInstanceError):
        r._policies={}
    with pytest.raises(RiskPolicyUnavailable):
        r.get("unknown")
    with pytest.raises(RiskPolicyUnavailable):
        ResearchRiskPolicyRegistry(policies[:1],registry)
    for field in ["model_sha256","dataset_sha256","manifest_sha256"]:
        changed=replace(info[0],**{field:"e"*64})
        bad=ModelRegistry([SimpleNamespace(info=changed),SimpleNamespace(info=info[1])])
        with pytest.raises(RiskPolicyUnavailable):
            ResearchRiskPolicyRegistry(policies,bad)


def test_loader_pin_corruption_and_missing(raw_policies,tmp_path):
    import hashlib
    registry=ModelRegistry(SimpleNamespace(info=info_for(parsed(p))) for p in raw_policies)
    path=tmp_path/"policy.json";path.write_bytes(POLICY_PATH.read_bytes())
    pin=hashlib.sha256(path.read_bytes()).hexdigest()
    settings=ApiSettings(research_risk_policy_path=path,research_risk_policy_sha256=pin)
    assert len(ResearchRiskPolicyRegistry.load(settings,registry).list_policies())==2
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(RiskPolicyUnavailable) as e:
        ResearchRiskPolicyRegistry.load(settings,registry)
    assert e.value.reason=="hash_mismatch"
    path.unlink()
    with pytest.raises(RiskPolicyUnavailable) as e:
        ResearchRiskPolicyRegistry.load(settings,registry)
    assert e.value.reason=="missing_policy"


def test_quantiles_degenerate_invalid_and_deterministic():
    np=pytest.importorskip("numpy")
    from depi_ml.analysis.research_risk import thresholds
    values=np.linspace(.01,.99,3000)
    assert thresholds(values)==thresholds(values[::-1])
    for p in [[],[.4]*20,[0]*20,[1]*20,[np.nan,.2],[np.inf,.2],[-.1,.2],[.2,1.1],["0.2","0.3"],[True,False]]:
        with pytest.raises(ValueError):
            thresholds(p)


def test_float32_offline_boundary_matches_serving_exact_decimal():
    np=pytest.importorskip("numpy")
    from depi_ml.analysis.research_risk import assignments
    from depi_ml.domain.research_risk_policy import band_index
    raw=np.array([0.4351593256],dtype="float32")
    threshold=round(float(raw[0]),12)
    assert float(raw[0])<threshold
    assert assignments(raw,.2,threshold).tolist()==[band_index(float(raw[0]),.2,threshold)]==[1]


def test_19_predictors_remain_immutable_contract():
    from depi_ml.api.schemas.features import PredictionFeatures
    expected=["age_at_booking","client_sex","booking_lead_days","appointment_month","appointment_weekday",
              "appointment_hour","duration_minutes","is_fwa","clinic_id","scheduled_service_lines",
              "distinct_body_areas","single_body_area_id","has_medical_evaluation","has_type4_service",
              "previous_attended","previous_no_show","previous_no_show_rate","days_since_previous_attended","days_since_previous_no_show"]
    assert PREDICTOR_COLUMNS==expected==list(PredictionFeatures.model_fields)
