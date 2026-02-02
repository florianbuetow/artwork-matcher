"""
FastAPI application factory.

Creates and configures the FastAPI application instance.
"""

from __future__ import annotations

from fastapi import FastAPI

from search_service.config import get_settings
from search_service.core.exceptions import register_exception_handlers
from search_service.core.lifespan import lifespan
from search_service.routers import health, index, info, search


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    # Load settings (validates configuration)
    settings = get_settings()

    # Create app with lifespan management
    application = FastAPI(
        title=f"{settings.service.name} Service",
        version=settings.service.version,
        lifespan=lifespan,
    )

    # Register exception handlers
    register_exception_handlers(application)

    # Register routers
    application.include_router(health.router, tags=["Operations"])
    application.include_router(info.router, tags=["Operations"])
    application.include_router(search.router, tags=["Search"])
    application.include_router(index.router, tags=["Index Management"])

    return application
