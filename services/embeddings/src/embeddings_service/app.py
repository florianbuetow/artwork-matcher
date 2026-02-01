"""
FastAPI application factory.

Creates and configures the FastAPI application instance.
"""

from __future__ import annotations

from fastapi import FastAPI

from embeddings_service.config import get_settings
from embeddings_service.core.exceptions import register_exception_handlers
from embeddings_service.core.lifespan import lifespan
from embeddings_service.routers import embed, health, info


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
        version=settings.service.version,
        lifespan=lifespan,
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(health.router, tags=["Operations"])
    app.include_router(info.router, tags=["Operations"])
    app.include_router(embed.router, tags=["Embeddings"])

    return app
