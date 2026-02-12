"""
Application lifecycle management.

Handles startup (client initialization) and shutdown (cleanup) events
for proper resource management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

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
        retry_max_attempts=settings.backends.retry.max_attempts,
        retry_initial_backoff_seconds=settings.backends.retry.initial_backoff_seconds,
        retry_max_backoff_seconds=settings.backends.retry.max_backoff_seconds,
        retry_jitter_seconds=settings.backends.retry.jitter_seconds,
        circuit_breaker_failure_threshold=settings.backends.circuit_breaker.failure_threshold,
        circuit_breaker_recovery_timeout_seconds=(
            settings.backends.circuit_breaker.recovery_timeout_seconds
        ),
    )

    state.search_client = SearchClient(
        base_url=settings.backends.search_url,
        timeout=settings.backends.timeout_seconds,
        service_name="search",
        retry_max_attempts=settings.backends.retry.max_attempts,
        retry_initial_backoff_seconds=settings.backends.retry.initial_backoff_seconds,
        retry_max_backoff_seconds=settings.backends.retry.max_backoff_seconds,
        retry_jitter_seconds=settings.backends.retry.jitter_seconds,
        circuit_breaker_failure_threshold=settings.backends.circuit_breaker.failure_threshold,
        circuit_breaker_recovery_timeout_seconds=(
            settings.backends.circuit_breaker.recovery_timeout_seconds
        ),
    )

    state.geometric_client = GeometricClient(
        base_url=settings.backends.geometric_url,
        timeout=settings.backends.timeout_seconds,
        service_name="geometric",
        retry_max_attempts=settings.backends.retry.max_attempts,
        retry_initial_backoff_seconds=settings.backends.retry.initial_backoff_seconds,
        retry_max_backoff_seconds=settings.backends.retry.max_backoff_seconds,
        retry_jitter_seconds=settings.backends.retry.jitter_seconds,
        circuit_breaker_failure_threshold=settings.backends.circuit_breaker.failure_threshold,
        circuit_breaker_recovery_timeout_seconds=(
            settings.backends.circuit_breaker.recovery_timeout_seconds
        ),
    )

    # Check backend health (non-blocking, just log status)
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
