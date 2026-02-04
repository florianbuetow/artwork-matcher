"""
Structured JSON logging for production observability.

Features:
- JSON format for log aggregation platforms
- Timestamp format: yyyy-mm-dd hh:mm
- Log level controlled via configuration
- Consistent field structure across all services
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON.

    Format:
    {
        "timestamp": "2025-01-15 10:30",
        "level": "INFO",
        "logger": "geometric_service",
        "message": "Server started",
        "extra": { ... }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # Timestamp in yyyy-mm-dd hh:mm format
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%d %H:%M")

        log_data: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include extra fields from record
        # Skip standard LogRecord attributes
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

        extra = {key: value for key, value in record.__dict__.items() if key not in standard_attrs}

        if extra:
            log_data["extra"] = extra

        return json.dumps(log_data, default=str)


VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def setup_logging(level: str, service_name: str) -> logging.Logger:
    """
    Configure structured JSON logging for the service.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service for logger identification

    Returns:
        Configured logger instance
    """
    # Validate log level
    level_upper = level.upper()
    if level_upper not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {sorted(VALID_LOG_LEVELS)}")
    numeric_level = getattr(logging, level_upper)

    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create JSON handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__ of calling module)

    Returns:
        Logger instance
    """
    # Import here to avoid circular dependency (config imports logging)
    # nosemgrep: config.semgrep.python.no-noqa-for-typing
    from geometric_service.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    base_name = settings.service.name
    full_name = f"{base_name}.{name}"

    return logging.getLogger(full_name)
