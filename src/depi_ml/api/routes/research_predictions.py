"""v2 clasifica resultados del mismo motor v1; sin inferencia duplicada ni I/O."""

from fastapi import APIRouter, Depends, Request

from depi_ml.api.dependencies import prediction_use_case, comparison_use_case
from depi_ml.api.mappers.prediction_mapper import prediction_response
from depi_ml.api.routes.predictions import to_input
from depi_ml.api.schemas.errors import ERROR_RESPONSES
from depi_ml.api.schemas.prediction import PredictionRequest, CompareRequest, Comparison
from depi_ml.api.schemas.research_risk import (
    ExperimentalRisk, ResearchPredictionResponse, ResearchCompareResponse,
    ResearchPolicyCatalog, ResearchPolicyView,
)
from depi_ml.api.security import require_service_key
from depi_ml.domain.research_risk_policy import ResearchRiskClassifier, RiskPolicyUnavailable

router=APIRouter(prefix="/internal/v2",tags=["research-risk"],dependencies=[Depends(require_service_key)])
LABELS={"LOW":"Bajo (exploratorio)","MEDIUM":"Medio (exploratorio)","HIGH":"Alto (exploratorio)"}


def require_research_policy(request: Request):
    policies=request.app.state.research_risk_policies
    if policies is None:
        raise RiskPolicyUnavailable(request.app.state.research_risk_reason)
    return policies


def research_response(result,policies,request_id):
    policy=policies.get(result.info.model_key)
    level=ResearchRiskClassifier.classify(result.probability_no_show,policy,result.info)
    legacy=prediction_response(result,request_id).model_dump(exclude={"api_version"})
    return ResearchPredictionResponse(**legacy,model_hash_sha256=result.info.model_sha256,
        probability_no_show_percent=result.probability_no_show*100,
        experimental_risk=ExperimentalRisk(level=level,label=LABELS[level],policy_id=policy.policy_id,
            policy_version=policy.policy_version,policy_hash=policy.policy_hash_sha256,
            low_upper_threshold=policy.low_upper_threshold,high_lower_threshold=policy.high_lower_threshold,
            boundary_rule=policy.boundary_rule,warnings=policy.warnings))


@router.get("/risk-policy",response_model=ResearchPolicyCatalog,responses=ERROR_RESPONSES,
            operation_id="ResearchRiskPolicyCatalog",description="Bandas relativas provisionales; nunca aprobación operativa. UNAVAILABLE no afecta v1.")
def policy_catalog(request: Request):
    policies=request.app.state.research_risk_policies
    views=[]
    if policies is not None:
        for p in policies.list_policies():
            views.append(ResearchPolicyView(model_key=p.model_key,model_version=p.model_version,
                model_hash_sha256=p.model_sha256,policy_id=p.policy_id,policy_version=p.policy_version,
                policy_hash=p.policy_hash_sha256,low_upper_threshold=p.low_upper_threshold,
                high_lower_threshold=p.high_lower_threshold,boundary_rule=p.boundary_rule,method=p.method,
                meaning=p.meaning,development_start=p.development_start,development_end_exclusive=p.development_end_exclusive,
                label_cutoff_exclusive=p.label_cutoff_exclusive,development_rows=p.development_rows,
                calibration_ece_uniform_10=p.calibration_ece,warnings=p.warnings))
    return ResearchPolicyCatalog(status="RESEARCH_PROVISIONAL" if policies is not None else "UNAVAILABLE",
        available=policies is not None,reason=request.app.state.research_risk_reason,
        policies=tuple(views),request_id=request.state.request_id)


@router.post("/predictions",response_model=ResearchPredictionResponse,responses=ERROR_RESPONSES,
             operation_id="ResearchPredictNoShow",description="Mismo score crudo v1, porcentaje y banda relativa por modelo; operational_use_allowed=false.")
def research_single(body: PredictionRequest,request: Request,
                    policies=Depends(require_research_policy),use_case=Depends(prediction_use_case)):
    result=use_case.execute(to_input(body,body.model_key))
    return research_response(result,policies,request.state.request_id)


@router.post("/predictions/compare",response_model=ResearchCompareResponse,responses=ERROR_RESPONSES,
             operation_id="ResearchCompareModels",description="RF y XGB independientes; sin winner, ensemble ni banda combinada.")
def research_compare(body: CompareRequest,request: Request,
                     policies=Depends(require_research_policy),use_case=Depends(comparison_use_case)):
    results=tuple(research_response(r,policies,request.state.request_id) for r in use_case.execute(to_input(body)))
    return ResearchCompareResponse(results=results,request_id=request.state.request_id,
        comparison=Comparison(absolute_probability_difference=abs(results[0].probability_no_show-results[1].probability_no_show)))
