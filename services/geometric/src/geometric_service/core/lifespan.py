"""Application lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from geometric_service.config import get_settings
from geometric_service.core.state import init_app_state
from geometric_service.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle."""
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

    # Log algorithm configuration
    logger.info(
        "Algorithm configuration",
        extra={
            "orb_max_features": settings.orb.max_features,
            "ratio_threshold": settings.matching.ratio_threshold,
            "ransac_reproj_threshold": settings.ransac.reproj_threshold,
            "min_inliers": settings.verification.min_inliers,
        },
    )

    # nosemgrep: python.lang.maintainability.useless-ifelse.useless-if-body
    logger.info("Service ready to accept requests")

    yield  # Application runs here

    # === SHUTDOWN ===
    # nosemgrep: python.lang.maintainability.useless-ifelse.useless-if-body
    logger.info(
        "Service shutting down",
        extra={
            "uptime_seconds": state.uptime_seconds,
            "uptime": state.uptime_formatted,
        },
    )
