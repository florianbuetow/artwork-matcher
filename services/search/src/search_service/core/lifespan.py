"""
Application lifecycle management.

Handles startup (index initialization and optional loading) and shutdown events
for proper resource management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from search_service.config import Settings, get_settings
from search_service.core.state import init_app_state
from search_service.logging import get_logger, setup_logging
from search_service.services.faiss_index import FAISSIndex, IndexLoadError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def create_faiss_index(settings: Settings) -> FAISSIndex:
    """
    Create a new FAISS index with configured dimension.

    Args:
        settings: Application settings

    Returns:
        New FAISSIndex instance
    """
    return FAISSIndex(dimension=settings.faiss.embedding_dimension)


def try_load_index(faiss_index: FAISSIndex, settings: Settings) -> tuple[bool, str | None]:
    """
    Attempt to load index from disk if auto_load is enabled and files exist.

    Args:
        faiss_index: The FAISS index wrapper to load into
        settings: Application settings

    Returns:
        Tuple of (success, error_message). error_message is None on success
        or if auto_load is disabled or files don't exist. error_message is
        set only when files exist but loading fails (corruption, etc.).
    """
    logger = get_logger()

    if not settings.index.auto_load:
        logger.info("Auto-load disabled, starting with empty index")
        return False, None

    index_path = Path(settings.index.path)
    metadata_path = Path(settings.index.metadata_path)

    if not index_path.exists():
        logger.info(
            "Index file not found, starting with empty index",
            extra={"index_path": str(index_path)},
        )
        return False, None

    if not metadata_path.exists():
        logger.info(
            "Metadata file not found, starting with empty index",
            extra={"metadata_path": str(metadata_path)},
        )
        return False, None

    try:
        faiss_index.load(index_path, metadata_path)
        logger.info(
            "Loaded index from disk",
            extra={
                "index_path": str(index_path),
                "count": faiss_index.count,
            },
        )
        return True, None
    except IndexLoadError as e:
        error_message = str(e)
        logger.warning(
            "Failed to load index, starting with empty index",
            extra={
                "error": error_message,
                "index_path": str(index_path),
            },
        )
        return False, error_message


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle.

    Startup:
    - Initialize logging
    - Initialize application state
    - Create FAISS index
    - Auto-load index if configured and files exist
    - Log startup information

    Shutdown:
    - Log shutdown with uptime
    """
    # === STARTUP ===
    settings = get_settings()

    # Initialize logging first
    setup_logging(settings.logging.level, settings.service.name)
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

    # Create FAISS index
    logger.info(
        "Initializing FAISS index",
        extra={
            "dimension": settings.faiss.embedding_dimension,
            "index_type": settings.faiss.index_type,
            "metric": settings.faiss.metric,
        },
    )

    faiss_index = create_faiss_index(settings)
    state.faiss_index = faiss_index

    # Try to auto-load index
    _loaded, load_error = try_load_index(faiss_index, settings)
    state.index_load_error = load_error

    logger.info(
        "Service ready to accept requests",
        extra={
            "index_count": faiss_index.count,
            "index_loaded": not faiss_index.is_empty,
        },
    )

    yield  # Application runs here

    # === SHUTDOWN ===
    logger.info(
        "Service shutting down",
        extra={
            "uptime_seconds": state.uptime_seconds,
            "uptime": state.uptime_formatted,
            "final_index_count": faiss_index.count,
        },
    )
