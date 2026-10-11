"""Factory ASGI de demo interna; cero carga ML o acceso DB durante importación."""

from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI, Depends

from depi_ml.api.exception_handlers import register_exception_handlers
from depi_ml.api.routes import health, models, predictions
from depi_ml.api.routes import prediction_explanations, prediction_batches
from depi_ml.application.use_cases.predict_no_show import PredictNoShow
from depi_ml.application.use_cases.compare_models import CompareModels
from depi_ml.application.use_cases.explain_prediction import ExplainPrediction
from depi_ml.application.use_cases.predict_batch import PredictBatch
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.prediction_engine import PredictionEngine
from depi_ml.api.middleware import RequestGuard
from depi_ml.api.routes import research_predictions
from depi_ml.api.openapi_versions import versioned_openapi
from depi_ml.domain.research_risk_policy import RiskPolicyUnavailable
from depi_ml.infrastructure.ml.research_risk_registry import ResearchRiskPolicyRegistry
from depi_ml.api.security import require_service_key

logger = logging.getLogger(__name__)


def create_app(settings: ApiSettings | None = None, *, registry_loader=None, explainer_loader=None, engine_factory=None,
               risk_policy_loader=None):
    settings = settings if settings is not None else ApiSettings.from_env()
    loader = registry_loader or ModelRegistry.load

    @asynccontextmanager
    async def lifespan(app):
        app.state.registry = app.state.predict_no_show = app.state.compare_models = None
        app.state.explain_prediction = app.state.predict_batch = None
        app.state.research_risk_policies = None
        app.state.research_risk_reason = "disabled" if not settings.enable_research_risk else "models_unavailable"
        started = perf_counter()
        try:
            if settings.demo_enabled:
                registry = loader(settings)
                if not registry.ready:
                    raise ArtifactVerificationError("both_models_required")
                explainers = {}
                if settings.enable_local_explanations:
                    try:
                        if explainer_loader is None:
                            from depi_ml.infrastructure.ml.shap_explainer import load_explainers
                            explainers = load_explainers(registry)
                        else:
                            explainers = explainer_loader(registry)
                    except Exception:
                        logger.warning("event=explanation_unavailable reason=initialization_failed")
                engine = (engine_factory or PredictionEngine)(registry, settings.max_concurrent_predictions, explainers)
                app.state.predict_no_show = PredictNoShow(engine, settings.default_model_key)
                app.state.compare_models = CompareModels(engine)
                app.state.explain_prediction = ExplainPrediction(engine, settings.default_model_key)
                app.state.predict_batch = PredictBatch(engine, settings.max_batch_items, settings.default_model_key)
                app.state.registry = registry
                if settings.enable_research_risk:
                    try:
                        app.state.research_risk_policies = (risk_policy_loader or ResearchRiskPolicyRegistry.load)(settings,registry)
                        app.state.research_risk_reason = None
                    except RiskPolicyUnavailable as error:
                        app.state.research_risk_reason = error.reason
                        logger.warning("event=research_risk_unavailable reason=%s",error.reason)
                    except Exception:
                        app.state.research_risk_reason = "invalid_policy"
                        logger.warning("event=research_risk_unavailable reason=invalid_policy")
                logger.info("event=registry_ready elapsed_ms=%.3f operational_use_allowed=false",
                            (perf_counter() - started) * 1000)
            else:
                logger.warning("event=registry_unavailable reason=experimental_demo_disabled")
        except ArtifactVerificationError as error:
            logger.error("event=registry_unavailable reason=%s", error.reason)
        except Exception:
            logger.error("event=registry_unavailable reason=startup_failed")
        try:
            yield
        finally:
            app.state.registry = app.state.predict_no_show = app.state.compare_models = None
            app.state.explain_prediction = app.state.predict_batch = None
            app.state.research_risk_policies = None
            app.state.research_risk_reason = "models_unavailable"

    docs = settings.expose_docs and settings.env == "development"
    app = FastAPI(title="DEPI internal ML demo", version="v1", lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if docs else None)
    app.state.settings = settings
    app.state.registry = app.state.predict_no_show = app.state.compare_models = None
    app.state.explain_prediction = app.state.predict_batch = None
    app.state.research_risk_policies = None
    app.state.research_risk_reason = "disabled"
    register_exception_handlers(app)
    app.add_middleware(RequestGuard, settings=settings)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predictions.router)
    app.include_router(prediction_explanations.router)
    app.include_router(prediction_batches.router)
    app.state.openapi_v1_routes = tuple(app.routes)
    app.include_router(research_predictions.router)
    app.openapi = lambda: versioned_openapi(app,"v1")
    if docs:
        @app.get("/internal/v2/openapi.json",include_in_schema=False,
                 dependencies=[Depends(require_service_key)])
        def research_openapi():
            return versioned_openapi(app,"v2")
    return app
