"""
FastAPI application factory.

Creates and configures the FastAPI application instance.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from gateway.config import get_settings
from gateway.core.exceptions import register_exception_handlers
from gateway.core.lifespan import lifespan
from gateway.routers import health, identify, info, objects


def serve_frontend() -> FileResponse:
    """Serve static frontend entrypoint."""
    static_index_path = Path(__file__).resolve().parents[2] / "static" / "index.html"
    return FileResponse(path=static_index_path)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    # Load settings (validates configuration)
    settings = get_settings()

    # Create app with lifespan management
    app = FastAPI(
        title=f"{settings.service.name} Service",
        description="API gateway for artwork identification",
        version=settings.service.version,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(health.router, tags=["Operations"])
    app.include_router(info.router, tags=["Operations"])
    app.include_router(identify.router, tags=["Identification"])
    app.include_router(objects.router, tags=["Objects"])
    app.add_api_route("/", serve_frontend, methods=["GET"], include_in_schema=False)

    return app


# nosemgrep: no-module-level-constants (required for uvicorn to load the app)
app = create_app()
