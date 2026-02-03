"""
FastAPI application factory.
"""

from __future__ import annotations

from fastapi import FastAPI

from geometric_service.config import get_settings
from geometric_service.core.exceptions import register_exception_handlers
from geometric_service.core.lifespan import lifespan
from geometric_service.routers import extract, health, info, match


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance with all routers registered
    """
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.service.name} Service",
        version=settings.service.version,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(health.router, tags=["Operations"])
    app.include_router(info.router, tags=["Operations"])
    app.include_router(extract.router, tags=["Features"])
    app.include_router(match.router, tags=["Matching"])

    return app
