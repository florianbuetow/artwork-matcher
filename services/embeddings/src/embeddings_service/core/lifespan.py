"""
Application lifecycle management.

Handles startup (model loading) and shutdown (cleanup) events
for proper resource management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoImageProcessor, AutoModel

from embeddings_service.config import Settings, get_config_path, get_settings
from embeddings_service.core.state import init_app_state
from embeddings_service.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def get_device(device_config: str) -> torch.device:
    """
    Resolve device from config string.

    Args:
        device_config: One of "auto", "cpu", "cuda", "mps"

    Returns:
        Resolved torch device
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


def get_cache_dir(settings: Settings) -> Path | None:
    """
    Get the cache directory for model weights.

    Args:
        settings: Application settings

    Returns:
        Absolute path to cache directory, or None if not configured
    """
    cache_dir_str = settings.model.cache_dir
    if not cache_dir_str:
        return None

    cache_dir = Path(cache_dir_str)

    # If relative path, make it relative to the config file location
    if not cache_dir.is_absolute():
        config_path = get_config_path()
        cache_dir = config_path.parent / cache_dir

    # Ensure directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def load_model(
    settings: Settings,
) -> tuple[AutoModel, AutoImageProcessor, torch.device]:
    """
    Load DINOv2 model and processor.

    Args:
        settings: Application settings

    Returns:
        Tuple of (model, processor, device)
    """
    device = get_device(settings.model.device)
    cache_dir = get_cache_dir(settings)

    # Convert to string for HuggingFace (it doesn't accept Path)
    cache_dir_str = str(cache_dir) if cache_dir else None

    # Load processor (handles image preprocessing)
    # nosemgrep: no-type-ignore
    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]  # nosec B615
        settings.model.name,
        cache_dir=cache_dir_str,
        revision=settings.model.revision,
    )

    # Load model
    model = AutoModel.from_pretrained(  # nosec B615
        settings.model.name,
        cache_dir=cache_dir_str,
        revision=settings.model.revision,
    )

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    return model, processor, device


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle.

    Startup:
    - Initialize logging
    - Initialize application state
    - Load DINOv2 model
    - Log startup information

    Shutdown:
    - Log shutdown with uptime
    - Clean up resources
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

    # Load DINOv2 model
    logger.info(
        "Loading DINOv2 model",
        extra={
            "model": settings.model.name,
            "cache_dir": settings.model.cache_dir,
        },
    )

    try:
        model, processor, device = load_model(settings)
    except Exception:
        logger.exception(
            "Failed to load model",
            extra={
                "model": settings.model.name,
                "device": settings.model.device,
            },
        )
        raise

    state.model = model
    state.processor = processor
    state.device = device

    logger.info(
        "Model loaded successfully",
        extra={
            "device": str(device),
            "embedding_dimension": settings.model.embedding_dimension,
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

    # Clean up model to free memory
    del state.model
    del state.processor
