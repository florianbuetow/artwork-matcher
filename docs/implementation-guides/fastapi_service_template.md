# FastAPI Service Template for Artwork Matcher

This document defines the standard structure, configuration patterns, and boilerplate code for all microservices in the Artwork Matcher project. Every service must follow this template to ensure consistency, maintainability, and production readiness.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Directory Structure](#directory-structure)
3. [Configuration Management](#configuration-management)
4. [Structured JSON Logging](#structured-json-logging)
5. [Application Factory Pattern](#application-factory-pattern)
6. [Router Organization](#router-organization)
7. [Standard Endpoints](#standard-endpoints)
8. [Production Deployment](#production-deployment)
9. [Complete File Reference](#complete-file-reference)

---

## Design Principles

### Fail Fast, No Fallbacks

Every service adheres to these non-negotiable rules:

| Principle | Implementation |
|-----------|----------------|
| **No default values** | All config must be explicitly specified in YAML |
| **Fail fast** | Missing config crashes on startup with clear error |
| **No error masking** | Exceptions propagate with full context |
| **No silent fallbacks** | CI rules enforce this via semgrep |
| **Explicit over implicit** | Every behavior is configured, never assumed |

### Configuration Philosophy

```python
# ❌ FORBIDDEN - Never do this
class Config:
    port: int = 8000  # Default values are banned
    
# ✅ REQUIRED - Always do this
class Config:
    port: int  # Must be specified in config.yaml or startup fails
```

---

## Directory Structure

```
services/<service_name>/
├── src/<service_name>/
│   ├── __init__.py
│   ├── main.py              # Entry point with main() function
│   ├── app.py               # FastAPI app factory
│   ├── config.py            # Pydantic settings + YAML loading
│   ├── logging.py           # Structured JSON logging setup
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lifespan.py      # Startup/shutdown lifecycle
│   │   ├── state.py         # Application state (uptime, etc.)
│   │   └── exceptions.py    # Custom exception handlers
│   └── routers/
│       ├── __init__.py
│       ├── health.py        # GET /health
│       ├── info.py          # GET /info
│       └── <domain>.py      # Service-specific routes
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── config.yaml              # Service configuration
├── pyproject.toml
├── justfile
├── Dockerfile
└── pyrightconfig.json
```

---

## Configuration Management

### config.yaml Template

```yaml
# Service Configuration
# ALL values are REQUIRED - no defaults exist in code

service:
  name: "embeddings"
  version: "0.1.0"

server:
  host: "0.0.0.0"
  port: 8000

logging:
  level: "INFO"    # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json"   # json (always json for production)

# Service-specific configuration sections below
# Example for embeddings service:
model:
  name: "facebook/dinov2-base"
  device: "auto"
  embedding_dimension: 768

preprocessing:
  image_size: 518
  normalize: true
```

### config.py - Configuration Module

```python
"""
Configuration management for the service.

Loads configuration from YAML with ZERO defaults.
Every value must be explicitly specified or startup fails.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ServiceConfig(BaseModel):
    """Service identity configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: str
    format: str


class Settings(BaseModel):
    """
    Root configuration container.

    All fields are REQUIRED. No defaults exist.
    Missing fields cause immediate startup failure.

    Usage:
        from <service_name>.config import get_settings
        settings = get_settings()
    """

    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig
    server: ServerConfig
    logging: LoggingConfig

    # Add service-specific config sections here
    # Example:
    # model: ModelConfig
    # preprocessing: PreprocessingConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigurationError: If file is missing, empty, or invalid
    """
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {config_path.absolute()}\n"
            f"Create the file or set CONFIG_PATH environment variable."
        )

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}") from e

    if config is None:
        raise ConfigurationError(
            f"Configuration file is empty: {config_path}\n"
            f"All configuration values must be explicitly specified."
        )

    return config


def get_config_path() -> Path:
    """
    Determine configuration file path.

    Priority:
    1. CONFIG_PATH environment variable
    2. ./config.yaml (relative to working directory)

    Returns:
        Path to configuration file
    """
    config_path_str = os.environ.get("CONFIG_PATH", "config.yaml")
    return Path(config_path_str)


@lru_cache
def get_settings() -> Settings:
    """
    Load and validate configuration.

    Cached to ensure single instance across application.
    Called once at startup - fails fast on invalid config.

    Returns:
        Validated Settings instance

    Raises:
        ConfigurationError: Config file missing or invalid
        ValidationError: Config values fail validation
    """
    config_path = get_config_path()
    yaml_config = load_yaml_config(config_path)

    try:
        return Settings(**yaml_config)
    except Exception as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}\n"
            f"All configuration values must be explicitly specified.\n"
            f"No default values are allowed."
        ) from e


def clear_settings_cache() -> None:
    """Clear the settings cache. Used in testing."""
    get_settings.cache_clear()
```

### Sensitive Data Filtering

```python
# Add to config.py

import re
from typing import Any


# Keywords that indicate sensitive data (case-insensitive)
SENSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "key",
    "secret",
    "pass",
    "password",
    "token",
    "credential",
    "auth",
    "api_key",
    "apikey",
    "private",
    "bearer",
})

# Compiled regex for efficient matching
_SENSITIVE_PATTERN = re.compile(
    r"(" + "|".join(re.escape(kw) for kw in SENSITIVE_KEYWORDS) + r")",
    re.IGNORECASE,
)


def is_sensitive_key(key: str) -> bool:
    """
    Check if a configuration key contains sensitive keywords.

    Args:
        key: Configuration key name

    Returns:
        True if the key likely contains sensitive data
    """
    return bool(_SENSITIVE_PATTERN.search(key))


def redact_sensitive_values(
    data: dict[str, Any],
    redaction_marker: str = "[REDACTED]",
) -> dict[str, Any]:
    """
    Recursively redact sensitive values from configuration.

    Used by /info endpoint to safely expose configuration.

    Args:
        data: Configuration dictionary
        redaction_marker: String to replace sensitive values

    Returns:
        New dictionary with sensitive values redacted
    """
    result: dict[str, Any] = {}

    for key, value in data.items():
        if is_sensitive_key(key):
            result[key] = redaction_marker
        elif isinstance(value, dict):
            result[key] = redact_sensitive_values(value, redaction_marker)
        elif isinstance(value, list):
            result[key] = [
                redact_sensitive_values(item, redaction_marker)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def get_safe_config() -> dict[str, Any]:
    """
    Get configuration with sensitive values redacted.

    Returns:
        Configuration dictionary safe for logging/API exposure
    """
    settings = get_settings()
    raw_config = settings.model_dump()
    return redact_sensitive_values(raw_config)
```

---

## Structured JSON Logging

### logging.py - Logging Module

```python
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
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON.

    Format:
    {
        "timestamp": "2025-01-15 10:30",
        "level": "INFO",
        "logger": "embeddings_service",
        "message": "Server started",
        "extra": { ... }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # Timestamp in yyyy-mm-dd hh:mm format
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M")

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

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs
        }

        if extra:
            log_data["extra"] = extra

        return json.dumps(log_data, default=str)


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
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

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


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__ of calling module)

    Returns:
        Logger instance
    """
    from .config import get_settings

    settings = get_settings()
    base_name = settings.service.name

    if name:
        full_name = f"{base_name}.{name}"
    else:
        full_name = base_name

    return logging.getLogger(full_name)
```

---

## Application Factory Pattern

### core/state.py - Application State

```python
"""
Application state management.

Tracks runtime state like uptime for health endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AppState:
    """
    Runtime application state.

    Attributes:
        start_time: When the application started (UTC)
    """

    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        now = datetime.now(timezone.utc)
        delta = now - self.start_time
        return delta.total_seconds()

    @property
    def uptime_formatted(self) -> str:
        """
        Format uptime as human-readable string.

        Returns:
            String like "2d 3h 15m 42s" or "15m 42s"
        """
        seconds = int(self.uptime_seconds)
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)


# Global application state instance
# Initialized in lifespan context
app_state: AppState | None = None


def get_app_state() -> AppState:
    """
    Get the current application state.

    Raises:
        RuntimeError: If called before app startup
    """
    if app_state is None:
        raise RuntimeError("Application state not initialized")
    return app_state


def init_app_state() -> AppState:
    """Initialize application state. Called during startup."""
    global app_state
    app_state = AppState()
    return app_state
```

### core/lifespan.py - Application Lifecycle

```python
"""
Application lifecycle management.

Handles startup and shutdown events for proper resource management.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import get_settings
from ..logging import get_logger, setup_logging
from .state import init_app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle.

    Startup:
    - Initialize logging
    - Initialize application state
    - Log startup information

    Shutdown:
    - Log shutdown
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

    # === SERVICE-SPECIFIC STARTUP ===
    # Add model loading, index loading, etc. here
    # Example:
    # await load_model(settings.model)

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

    # === SERVICE-SPECIFIC CLEANUP ===
    # Add resource cleanup here
    # Example:
    # await unload_model()
```

### core/exceptions.py - Exception Handlers

```python
"""
Custom exception handlers for consistent error responses.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import ConfigurationError
from ..logging import get_logger


class ServiceError(Exception):
    """
    Base exception for service errors.

    Attributes:
        error: Machine-readable error code
        message: Human-readable description
        status_code: HTTP status code
        details: Additional context
    """

    def __init__(
        self,
        error: str,
        message: str,
        status_code: int,
        details: dict[str, object] | None,
    ) -> None:
        self.error = error
        self.message = message
        self.status_code = status_code
        if details is None:
            self.details: dict[str, object] = {}
        else:
            self.details = details
        super().__init__(message)


async def service_error_handler(
    request: Request,
    exc: ServiceError,
) -> JSONResponse:
    """Handle ServiceError exceptions."""
    logger = get_logger()
    logger.warning(
        "Service error",
        extra={
            "error": exc.error,
            "message": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "message": exc.message,
            "details": exc.details,
        },
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Logs full traceback but returns sanitized error to client.
    """
    logger = get_logger()
    logger.exception(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "details": {},
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
```

### app.py - Application Factory

```python
"""
FastAPI application factory.

Creates and configures the FastAPI application instance.
"""

from __future__ import annotations

from fastapi import FastAPI

from .config import get_settings
from .core.exceptions import register_exception_handlers
from .core.lifespan import lifespan
from .routers import health, info


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    # Load settings (validates configuration)
    settings = get_settings()

    # Create app with lifespan management
    app = FastAPI(
        title=f"{settings.service.name} Service",
        version=settings.service.version,
        lifespan=lifespan,
        # Disable automatic docs in production if needed
        # docs_url=None,
        # redoc_url=None,
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(health.router, tags=["Operations"])
    app.include_router(info.router, tags=["Operations"])

    # Register service-specific routers here
    # Example:
    # from .routers import embeddings
    # app.include_router(embeddings.router, tags=["Embeddings"])

    return app
```

### main.py - Entry Point

```python
"""
Service entry point.

This module provides the main() function for running the service
with production-grade configuration.
"""

from __future__ import annotations

import sys

import uvicorn

from .app import create_app
from .config import ConfigurationError, get_settings


def main() -> int:
    """
    Run the service with uvicorn.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Validate configuration before starting
        settings = get_settings()
    except ConfigurationError as e:
        # Print to stderr - logging isn't configured yet
        print(f"FATAL: Configuration error\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FATAL: Unexpected error during configuration\n{e}", file=sys.stderr)
        return 1

    # Create application
    app = create_app()

    # Run with uvicorn
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        # Production settings
        log_level="warning",  # Uvicorn logs (our JSON logger handles app logs)
        access_log=False,  # Disable uvicorn access log (use middleware if needed)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Router Organization

### routers/__init__.py

```python
"""
API routers for the service.

Each router handles a specific domain of endpoints.
"""

from . import health, info

__all__ = ["health", "info"]
```

### routers/health.py - Health Endpoint

```python
"""
Health check endpoint.

Provides service health status for container orchestration
(Docker health checks, Kubernetes probes).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.state import get_app_state


router = APIRouter()


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: HealthStatus
    uptime_seconds: float
    uptime: str
    system_time: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns:
        Health status with uptime and system time
    """
    state = get_app_state()

    # System time in yyyy-mm-dd hh:mm format
    system_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    return HealthResponse(
        status=HealthStatus.HEALTHY,
        uptime_seconds=state.uptime_seconds,
        uptime=state.uptime_formatted,
        system_time=system_time,
    )
```

### routers/info.py - Info Endpoint

```python
"""
Service information endpoint.

Exposes service configuration and metadata.
Sensitive values are automatically redacted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import get_safe_config, get_settings


router = APIRouter()


class InfoResponse(BaseModel):
    """Service information response schema."""

    service: str
    version: str
    config: dict[str, Any]


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information and configuration.

    Sensitive values (keys, secrets, passwords) are automatically redacted.

    Returns:
        Service metadata and safe configuration
    """
    settings = get_settings()
    safe_config = get_safe_config()

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        config=safe_config,
    )
```

---

## Production Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies (no dev dependencies in production)
RUN uv sync --frozen --no-dev

# Copy application code
COPY config.yaml .
COPY src/ src/

# Set environment variables
ENV CONFIG_PATH=/app/config.yaml
ENV PYTHONUNBUFFERED=1

# Expose port (default, override in docker-compose)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with production-grade process manager
# Option 1: Direct uvicorn (simpler)
CMD ["uv", "run", "python", "-m", "<service_name>.main"]

# Option 2: With gunicorn for multi-worker (higher throughput)
# CMD ["uv", "run", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
#      "-w", "4", "-b", "0.0.0.0:8000", "<service_name>.app:create_app()"]
```

### docker-compose.yml Service Entry

```yaml
services:
  embeddings:
    build: ./services/embeddings
    ports:
      - "8001:8000"
    volumes:
      - ./data:/data:ro
    environment:
      - CONFIG_PATH=/app/config.yaml
    restart: unless-stopped  # Auto-restart on failure
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          memory: 4G
```

### Production Restart Mechanisms

For production-grade auto-restart, use one of these approaches:

#### Option 1: Docker Compose (Recommended for this project)

```yaml
# docker-compose.yml
services:
  embeddings:
    restart: unless-stopped  # Restarts on failure, not on manual stop
    # OR
    restart: always          # Always restart
    # OR  
    restart: on-failure      # Only restart on non-zero exit
```

#### Option 2: systemd (Linux servers)

```ini
# /etc/systemd/system/embeddings.service
[Unit]
Description=Embeddings Service
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/opt/artwork-matcher/services/embeddings
Environment=CONFIG_PATH=/opt/artwork-matcher/services/embeddings/config.yaml
ExecStart=/usr/bin/uv run python -m embeddings_service.main
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Option 3: Kubernetes (Enterprise)

```yaml
# k8s/embeddings-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embeddings
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: embeddings
          image: artwork-matcher/embeddings:latest
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            limits:
              memory: "4Gi"
              cpu: "2"
```

### justfile Run Commands

```makefile
# Run service in development (with reload)
run:
    @echo "\033[0;34m=== Running Service (Development) ===\033[0m"
    @uv run uvicorn <service_name>.app:create_app --factory --reload --host 0.0.0.0 --port 8001

# Run service in production mode (local testing)
run-prod:
    @echo "\033[0;34m=== Running Service (Production) ===\033[0m"
    @uv run python -m <service_name>.main
```

### justfile Kill Command

When using `uv run uvicorn --reload`, the process spawns multiple child processes that don't match simple `pgrep` patterns. Use port-based detection with `lsof` instead:

```makefile
# Stop the locally running service
kill:
    #!/usr/bin/env bash
    printf "\n"
    printf "\033[0;34m=== Stopping Service (Local) ===\033[0m\n"
    printf "\n"

    # Find process listening on the service port (e.g., 8001)
    pid=$(lsof -ti :8001 2>/dev/null)

    if [ -n "$pid" ]; then
        printf "Service is running (PID: %s). Stopping...\n" "$pid"
        kill $pid 2>/dev/null
        sleep 1

        # Check if still running
        if lsof -ti :8001 > /dev/null 2>&1; then
            printf "\033[0;31m✗ Service still running. Forcing kill...\033[0m\n"
            kill -9 $(lsof -ti :8001) 2>/dev/null
            sleep 1
        fi

        if lsof -ti :8001 > /dev/null 2>&1; then
            printf "\033[0;31m✗ Failed to stop service\033[0m\n"
            exit 1
        else
            printf "\033[0;32m✓ Service stopped\033[0m\n"
        fi
    else
        printf "Service is not running\n"
    fi
    printf "\n"
```

**Why port-based detection?**

The naive approach using `pgrep -f "uvicorn <service_name>"` fails because:
1. `uv run` wraps the command, changing the process name
2. `--reload` spawns a StatReload watcher + child worker processes
3. The actual process may appear as `python3.x` instead of `uvicorn`

Using `lsof -ti :PORT` reliably finds whatever process is listening on the port, regardless of how it was spawned.

**Note:** Use `printf` instead of `echo` for ANSI color codes in bash scripts, as `echo` doesn't interpret `\033` escape sequences on all systems.

---

## Complete File Reference

### pyproject.toml

```toml
[project]
name = "<service-name>"
version = "0.1.0"
description = "<Service description>"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "pydantic>=2.10.0",
    "pyyaml>=6.0.0",
    # Add service-specific dependencies
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pyright>=1.1.390",
    "bandit>=1.7.0",
    "deptry>=0.21.0",
    "codespell>=2.3.0",
    "semgrep>=1.99.0",
    "pip-audit>=2.7.0",
    "pygount>=1.8.0",
    "types-PyYAML>=6.0.0",
]

[project.scripts]
# Entry point for the service
<service-name> = "<service_name>.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/<service_name>"]

# === Ruff ===
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM", "TCH", "PTH", "ERA", "PL", "RUF",
]
ignore = ["PLR0913", "PLR2004"]

[tool.ruff.lint.isort]
known-first-party = ["<service_name>"]

# === Mypy ===
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
files = ["src"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# === Pytest ===
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
```

### Summary Checklist

When creating a new service, ensure you have:

**Configuration:**
- [ ] `config.yaml` with ALL required values (no defaults in code)
- [ ] `config.py` with Pydantic models (extra="forbid")
- [ ] Sensitive data redaction helper

**Logging:**
- [ ] `logging.py` with JSON formatter
- [ ] Timestamp format: `yyyy-mm-dd hh:mm`
- [ ] Log level from configuration

**Application:**
- [ ] `app.py` with create_app() factory
- [ ] `main.py` with main() entry point
- [ ] Lifespan context for startup/shutdown
- [ ] Application state tracking (uptime)

**Routers:**
- [ ] `/health` endpoint with uptime and system_time
- [ ] `/info` endpoint with safe (redacted) config
- [ ] Service-specific routers with APIRouter and tags

**Production:**
- [ ] Dockerfile with health check
- [ ] Docker Compose with restart policy
- [ ] justfile with run commands

---

## Testing Configuration

### tests/conftest.py

```python
"""Shared test fixtures."""

import pytest
from fastapi.testclient import TestClient

from <service_name>.app import create_app
from <service_name>.config import clear_settings_cache


@pytest.fixture(autouse=True)
def reset_settings():
    """Clear settings cache before each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)
```

### tests/test_health.py

```python
"""Tests for health endpoint."""


def test_health_returns_healthy(client):
    """Health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "uptime" in data
    assert "system_time" in data


def test_health_system_time_format(client):
    """System time is in correct format."""
    response = client.get("/health")
    data = response.json()
    # Format: yyyy-mm-dd hh:mm
    import re
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
    assert re.match(pattern, data["system_time"])
```

### tests/test_info.py

```python
"""Tests for info endpoint."""


def test_info_returns_config(client):
    """Info endpoint returns configuration."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "config" in data


def test_info_redacts_sensitive_values(client, monkeypatch):
    """Sensitive values are redacted in info response."""
    # This test would require injecting a config with sensitive keys
    # Implementation depends on your testing strategy
    pass
```

---

This template provides a solid foundation for all services in the Artwork Matcher project, ensuring consistency, maintainability, and production readiness.
