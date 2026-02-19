"""Application lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from storage_service.config import get_settings
from storage_service.core.state import init_app_state
from storage_service.logging import get_logger, setup_logging
from storage_service.services.blob_store import FileBlobStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle."""
    # === STARTUP ===
    settings = get_settings()

    # Initialize logging first
    setup_logging(settings.logging.level, settings.service.name)
    logger = get_logger(__name__)

    # Initialize application state
    state = init_app_state()

    # Create blob store
    store = FileBlobStore(root_dir=Path(settings.storage.path))
    state.blob_store = store

    logger.info(
        "Service starting",
        extra={
            "service": settings.service.name,
            "version": settings.service.version,
            "host": settings.server.host,
            "port": settings.server.port,
            "storage_path": settings.storage.path,
            "object_count": store.count(),
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
            "object_count": store.count(),
        },
    )
