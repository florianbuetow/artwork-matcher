"""
Structured JSON logging for production observability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from service_commons.logging import (
    VALID_LOG_LEVELS,
    JSONFormatter,
    get_named_logger,
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


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__ of calling module)

    Returns:
        Logger instance
    """
    # Import lazily to avoid import-time settings evaluation.
    # nosemgrep: config.semgrep.python.no-noqa-for-typing
    from geometric_service.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    return get_named_logger(settings.service.name, name)
