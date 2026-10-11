"""Carga JSON acotada/pineada una vez; sin archivos o datos históricos por request."""

from types import MappingProxyType
from dataclasses import dataclass
import hashlib
import json
import re

from depi_ml.domain.model_info import MODEL_KEYS
from depi_ml.domain.research_risk_policy import ResearchRiskPolicy, RiskPolicyUnavailable


@dataclass(frozen=True,init=False)
class ResearchRiskPolicyRegistry:
    _policies: object
    def __init__(self, policies, registry):
        entries=tuple(policies)
        mapping={p.model_key:p for p in entries}
        if len(entries)!=2 or set(mapping)!=set(MODEL_KEYS):
            raise RiskPolicyUnavailable("invalid_policy")
        for key,policy in mapping.items():
            info=registry.get(key).info
            if (policy.model_version!=info.model_version or policy.model_sha256!=info.model_sha256
                    or policy.feature_version!=info.feature_version or
                    policy.source.dataset_sha256!=info.dataset_sha256 or
                    policy.source.experiment_manifest_sha256!=info.manifest_sha256):
                raise RiskPolicyUnavailable("model_identity_mismatch")
        object.__setattr__(self,"_policies",MappingProxyType(mapping))

    def get(self, model_key):
        try:
            return self._policies[model_key]
        except KeyError:
            raise RiskPolicyUnavailable("missing_policy") from None

    def list_policies(self):
        return tuple(self.get(key) for key in MODEL_KEYS)

    @classmethod
    def load(cls,settings,registry):
        path,pin=settings.research_risk_policy_path,settings.research_risk_policy_sha256
        if path is None or pin is None:
            raise RiskPolicyUnavailable("missing_policy")
        if not re.fullmatch(r"[a-f0-9]{64}",pin):
            raise RiskPolicyUnavailable("invalid_policy")
        try:
            with path.open("rb") as stream:
                raw=stream.read(262145)
            if len(raw)>262144:
                raise RiskPolicyUnavailable("invalid_policy")
            if hashlib.sha256(raw).hexdigest()!=pin:
                raise RiskPolicyUnavailable("hash_mismatch")
            def unique_pairs(pairs):
                result={}
                for key,value in pairs:
                    if key in result:
                        raise ValueError("Duplicate policy key")
                    result[key]=value
                return result
            data=json.loads(raw,object_pairs_hook=unique_pairs)
            if (set(data)!={"artifact_type","format_version","policies"} or
                    data["artifact_type"]!="research_risk_policy" or type(data["format_version"]) is not int
                    or data["format_version"]!=1 or len(data["policies"])!=2):
                raise RiskPolicyUnavailable("invalid_policy")
            if any(p.get("support_passed") is not True for p in data["policies"]):
                raise RiskPolicyUnavailable("insufficient_support")
            policies=tuple(ResearchRiskPolicy.model_validate_json(json.dumps(p)) for p in data["policies"])
            if settings.model_reference_dir is not None:
                with (settings.model_reference_dir/"experiment.json").open("rb") as manifest_stream:
                    manifest_raw=manifest_stream.read(2*1024*1024+1)
                if len(manifest_raw)>2*1024*1024:
                    raise RiskPolicyUnavailable("invalid_policy")
                manifest_hash=hashlib.sha256(manifest_raw).hexdigest()
                if any(p.source.experiment_manifest_sha256!=manifest_hash for p in policies):
                    raise RiskPolicyUnavailable("model_identity_mismatch")
                source=json.loads(manifest_raw)["label_availability"]
                if any(p.source.labels_sha256!=source["sha256"] or
                       p.source.labels_manifest_sha256!=source["manifest_sha256"] for p in policies):
                    raise RiskPolicyUnavailable("model_identity_mismatch")
            return cls(policies,registry)
        except RiskPolicyUnavailable:
            raise
        except OSError:
            raise RiskPolicyUnavailable("missing_policy") from None
        except Exception:
            raise RiskPolicyUnavailable("invalid_policy") from None
