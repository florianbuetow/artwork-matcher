"""
Application lifecycle management.

Handles startup (client initialization) and shutdown (cleanup) events
for proper resource management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from gateway.clients import EmbeddingsClient, GeometricClient, SearchClient
from gateway.config import get_settings
from gateway.core.state import init_app_state
from gateway.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle.

    Startup:
    - Initialize logging
    - Initialize application state
    - Create backend HTTP clients
    - Log startup information

    Shutdown:
    - Log shutdown with uptime
    - Close HTTP clients
    """
    # === STARTUP ===
    settings = get_settings()

    # Initialize logging first
    setup_logging(settings.server.log_level, settings.service.name)
    logger = get_logger()

    # Initialize application state
    state = init_app_state()

    logger.info(
        "Service starting",
        extra={
            "service": settings.service.name,
            "version": settings.service.version,
            "host": settings.server.host,
            "port": settings.server.port,
        },
    )

    # Create backend clients
    logger.info(
        "Initializing backend clients",
        extra={
            "embeddings_url": settings.backends.embeddings_url,
            "search_url": settings.backends.search_url,
            "geometric_url": settings.backends.geometric_url,
            "timeout": settings.backends.timeout_seconds,
        },
    )

    state.embeddings_client = EmbeddingsClient(
        base_url=settings.backends.embeddings_url,
        timeout=settings.backends.timeout_seconds,
        service_name="embeddings",
    )

    state.search_client = SearchClient(
        base_url=settings.backends.search_url,
        timeout=settings.backends.timeout_seconds,
        service_name="search",
    )

    state.geometric_client = GeometricClient(
        base_url=settings.backends.geometric_url,
        timeout=settings.backends.timeout_seconds,
        service_name="geometric",
    )

    # Check backend health (non-blocking, just log status)
    try:
        embeddings_status = await state.embeddings_client.health_check()
        search_status = await state.search_client.health_check()
        geometric_status = await state.geometric_client.health_check()

        logger.info(
            "Backend health check completed",
            extra={
                "embeddings": embeddings_status,
                "search": search_status,
                "geometric": geometric_status,
            },
        )
    except httpx.HTTPError as e:
        logger.warning(
            "Backend health check failed during startup",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
    except Exception as e:
        logger.error(
            "Unexpected error during startup health check",
            extra={"error": str(e), "error_type": type(e).__name__},
            exc_info=True,
        )

    logger.info("Service ready to accept requests")

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info(
        "Service shutting down",
        extra={
            "uptime_seconds": state.uptime_seconds,
            "uptime": state.uptime_formatted,
        },
    )

    # Close HTTP clients
    if state.embeddings_client:
        await state.embeddings_client.close()
    if state.search_client:
        await state.search_client.close()
    if state.geometric_client:
        await state.geometric_client.close()

    logger.info("Service shutdown complete")
