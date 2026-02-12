"""
Structured JSON logging for production observability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from service_commons.logging import (
    VALID_LOG_LEVELS,
    JSONFormatter,
    get_service_logger,
    setup_logging,
)

if TYPE_CHECKING:
    import logging

__all__ = [
    "VALID_LOG_LEVELS",
    "JSONFormatter",
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
    from gateway.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    return get_service_logger(settings.service.name)
