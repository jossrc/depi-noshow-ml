"""Política research inmutable, identidad y fronteras; legacy RiskClassifier intacto."""

from datetime import datetime
import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from depi_ml.domain.errors import BasePredictionException

Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
ModelKey = Literal["random_forest_full_v1", "xgboost_full_v1"]
Level = Literal["LOW", "MEDIUM", "HIGH"]
LEVELS = ("LOW", "MEDIUM", "HIGH")
POLICY_VERSION = "research_risk_v1_1"


def band_index(probability,low,high):
    """Única regla de frontera, compartida por serving y evaluación offline."""
    if isinstance(probability,bool) or not isinstance(probability,(int,float)) or not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError("Invalid research probability")
    if not math.isfinite(low) or not math.isfinite(high) or not 0<low<high<1:
        raise ValueError("Invalid research thresholds")
    return 0 if probability<low else 1 if probability<high else 2


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class RiskPolicyUnavailable(BasePredictionException):
    code, status_code = "RESEARCH_RISK_POLICY_NOT_READY", 503
    message = "La política de bandas experimentales no está disponible."

    def __init__(self, reason="invalid_policy"):
        super().__init__(self.message)
        allowed = {"disabled", "models_unavailable", "missing_policy", "hash_mismatch",
                   "invalid_policy", "model_identity_mismatch", "insufficient_support"}
        self.reason = reason if reason in allowed else "invalid_policy"
        self.details = ({"field": "experimental_risk", "reason": self.reason},)


class FrozenPolicyObject(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class PolicySource(FrozenPolicyObject):
    dataset_sha256: Sha256
    dataset_manifest_sha256: Sha256
    labels_sha256: Sha256
    labels_manifest_sha256: Sha256
    experiment_manifest_sha256: Sha256


class BandEvidence(FrozenPolicyObject):
    model_key: ModelKey
    period: Literal["development_april_may", "diagnostic_june"]
    clinic_id: None
    level: Level
    rows: Annotated[int, Field(ge=100)]
    positives: Annotated[int, Field(ge=0)]
    observed_rate: Annotated[float, Field(ge=0,le=1)]
    coverage: Annotated[float, Field(ge=0,le=1)]
    clients: Annotated[int, Field(ge=20)]
    ci95: tuple[Finite, Finite]
    ci_method: Literal["client_cluster_percentile_1000_seed42_conditional_on_fixed_model_and_thresholds"]
    precision_if_band_selected: Finite
    recall_of_all_positive_events: Finite
    false_positives_if_band_selected: Annotated[int, Field(ge=0)]
    positive_events_outside_band: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def coherent_counts(self):
        if (self.positives>self.rows or self.clients>self.rows or
                not 0<=self.ci95[0]<=self.ci95[1]<=1 or
                abs(self.observed_rate-self.positives/self.rows)>1e-10 or
                self.false_positives_if_band_selected!=self.rows-self.positives):
            raise ValueError("Invalid band evidence")
        return self


class ResearchRiskPolicy(FrozenPolicyObject):
    policy_id: Annotated[str, Field(min_length=1,max_length=128,pattern=r"^[a-z0-9_]+$")]
    policy_version: Literal["research_risk_v1_1"]
    policy_hash_sha256: Sha256
    model_key: ModelKey
    model_version: str
    model_sha256: Sha256
    feature_version: Literal["BOOKING_BASELINE_V1"]
    prediction_target: Literal["NO_SHOW_CANDIDATE"]
    target_class: Literal[1]
    score_space: Literal["raw_probability_no_show"]
    low_upper_threshold: Finite
    high_lower_threshold: Finite
    boundary_rule: Literal["LOW:p<t_low;MEDIUM:t_low<=p<t_high;HIGH:p>=t_high"]
    method: Literal["score_quantiles_tertiles_exploratory"]
    quantile_interpolation: Literal["linear"]
    threshold_decimal_places: Literal[12]
    seed: Literal[42]
    development_start: str
    development_end_exclusive: str
    diagnostic_end_exclusive: str
    label_cutoff_exclusive: str
    source: PolicySource
    development_rows: Annotated[int, Field(ge=1000)]
    development_positives: Annotated[int, Field(gt=0)]
    development_prevalence: Finite
    band_evidence: tuple[BandEvidence, ...]
    calibration_ece: Annotated[float, Field(ge=0,le=1)]
    support_passed: Literal[True]
    status: Literal["RESEARCH_PROVISIONAL"]
    meaning: Literal["relative_score_band_not_validated_absolute_risk"]
    operational_use_allowed: Literal[False]
    approved_by: None
    warnings: tuple[Annotated[str, Field(min_length=1,max_length=512)], ...]

    @model_validator(mode="after")
    def verify_policy(self):
        if not 0<self.low_upper_threshold<self.high_lower_threshold<1:
            raise ValueError("Invalid research thresholds")
        if self.model_version!=f"{self.model_key}:{self.model_sha256[:12]}":
            raise ValueError("Invalid model version")
        times = [datetime.fromisoformat(t) for t in [self.development_start,self.development_end_exclusive,
                 self.diagnostic_end_exclusive,self.label_cutoff_exclusive]]
        if any(t.tzinfo is None for t in times) or not times[0]<times[1]<times[2] or times[2]!=times[3]:
            raise ValueError("Invalid research cohort boundaries")
        if self.development_positives>=self.development_rows or abs(
                self.development_prevalence-self.development_positives/self.development_rows)>1e-10:
            raise ValueError("Invalid development support")
        if len(self.band_evidence)!=6 or not self.warnings:
            raise ValueError("Missing research evidence")
        for period in ("development_april_may","diagnostic_june"):
            cells = [c for c in self.band_evidence if c.period==period]
            if ([c.level for c in cells]!=["LOW","MEDIUM","HIGH"] or
                    any(c.model_key!=self.model_key for c in cells) or
                    not cells[0].observed_rate<cells[1].observed_rate<cells[2].observed_rate):
                raise ValueError("Insufficient band separation")
            if period=="development_april_may" and (sum(c.rows for c in cells)!=self.development_rows or
                    sum(c.positives for c in cells)!=self.development_positives):
                raise ValueError("Inconsistent development cohort")
        value=self.model_dump(mode="json",exclude={"policy_hash_sha256"})
        if canonical_hash(value)!=self.policy_hash_sha256:
            raise ValueError("Policy content hash mismatch")
        return self


class ResearchRiskClassifier:
    @staticmethod
    def classify(probability, policy: ResearchRiskPolicy, info):
        if (info.model_key!=policy.model_key or info.model_version!=policy.model_version or
                info.model_sha256!=policy.model_sha256 or info.feature_version!=policy.feature_version):
            raise RiskPolicyUnavailable("model_identity_mismatch")
        return LEVELS[band_index(probability,policy.low_upper_threshold,policy.high_lower_threshold)]
