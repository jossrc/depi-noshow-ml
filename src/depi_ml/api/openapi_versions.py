"""Documentos separados: v1 idéntico al snapshot, v2 publicado aditivamente."""

from fastapi.openapi.utils import get_openapi
from depi_ml.api.routes.research_predictions import router as research_router


def versioned_openapi(app, version="v1"):
    # Capture the v1 route collection before adding research; works with lazy
    # included routers as well as concrete APIRoutes, without private internals.
    routes=app.state.openapi_v1_routes if version=="v1" else research_router.routes
    return get_openapi(title=app.title if version=="v1" else "DEPI internal ML research risk",
                       version=version,routes=routes)
