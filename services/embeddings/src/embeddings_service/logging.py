"""
Structured JSON logging for production observability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from service_commons.logging import (
    get_service_logger,
    setup_logging,
)

if TYPE_CHECKING:
    import logging

__all__ = [
    "get_logger",
    "setup_logging",
]


def get_logger() -> logging.Logger:
    """
    Get the service logger instance.

    Returns:
        Logger instance for the service
    """
    # Import lazily to avoid import-time settings evaluation.
    # nosemgrep: config.semgrep.python.no-noqa-for-typing
    from embeddings_service.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    return get_service_logger(settings.service.name)
