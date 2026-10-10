"""Factory ASGI de demo interna; cero carga ML o acceso DB durante importación."""

from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI

from depi_ml.api.exception_handlers import register_exception_handlers
from depi_ml.api.routes import health, models, predictions
from depi_ml.application.use_cases.predict_no_show import PredictNoShow
from depi_ml.application.use_cases.compare_models import CompareModels
from depi_ml.infrastructure.config.api_settings import ApiSettings
from depi_ml.infrastructure.ml.artifact_verifier import ArtifactVerificationError
from depi_ml.infrastructure.ml.model_registry import ModelRegistry
from depi_ml.infrastructure.ml.prediction_engine import PredictionEngine
from depi_ml.api.middleware import RequestGuard

logger = logging.getLogger(__name__)


def create_app(settings: ApiSettings | None = None, *, registry_loader=None):
    settings = settings if settings is not None else ApiSettings.from_env()
    loader = registry_loader or ModelRegistry.load

    @asynccontextmanager
    async def lifespan(app):
        app.state.registry = app.state.predict_no_show = app.state.compare_models = None
        started = perf_counter()
        try:
            if settings.demo_enabled:
                registry = loader(settings)
                if not registry.ready:
                    raise ArtifactVerificationError("both_models_required")
                engine = PredictionEngine(registry, settings.max_concurrent_predictions)
                app.state.predict_no_show = PredictNoShow(engine, settings.default_model_key)
                app.state.compare_models = CompareModels(engine)
                app.state.registry = registry
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

    docs = settings.expose_docs and settings.env == "development"
    app = FastAPI(title="DEPI internal ML demo", version="v1", lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if docs else None)
    app.state.settings = settings
    app.state.registry = app.state.predict_no_show = app.state.compare_models = None
    register_exception_handlers(app)
    app.add_middleware(RequestGuard, settings=settings)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(predictions.router)
    return app
