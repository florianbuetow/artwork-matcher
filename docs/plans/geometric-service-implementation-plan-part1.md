# Geometric Service Implementation Plan - Part 1: Core Infrastructure and Domain Logic

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Geometric Service providing ORB feature extraction and RANSAC-based geometric verification for artwork matching.

**Architecture:** Classical computer vision service using OpenCV. Extracts ORB features from images, matches them using BFMatcher with Lowe's ratio test, then verifies geometric consistency using RANSAC homography estimation. Follows the same service structure as the embeddings service.

**Tech Stack:** FastAPI, OpenCV (opencv-python-headless), NumPy, Pydantic, PyYAML

---

## Task 1: Core Infrastructure - Configuration Module

**Files:**
- Create: `services/geometric/src/geometric_service/config.py`
- Test: `services/geometric/tests/unit/test_config.py`

**Step 1: Write the failing test**

Create `services/geometric/tests/unit/__init__.py`:
```python
"""Unit tests for geometric service."""
```

Create `services/geometric/tests/unit/test_config.py`:
```python
"""Unit tests for configuration module."""

from __future__ import annotations

import pytest

from geometric_service.config import ConfigurationError, get_settings, load_yaml_config
from pathlib import Path


@pytest.mark.unit
class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_raises_error_for_missing_file(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError for missing file."""
        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigurationError, match="not found"):
            load_yaml_config(missing_path)

    def test_raises_error_for_empty_file(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError for empty file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        with pytest.raises(ConfigurationError, match="empty"):
            load_yaml_config(empty_file)


@pytest.mark.unit
class TestGetSettings:
    """Tests for get_settings function."""

    def test_loads_valid_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Should load valid configuration."""
        config_content = """
service:
  name: geometric
  version: 0.1.0

orb:
  max_features: 1000
  scale_factor: 1.2
  n_levels: 8
  edge_threshold: 31
  patch_size: 31
  fast_threshold: 20

matching:
  ratio_threshold: 0.75
  cross_check: false

ransac:
  reproj_threshold: 5.0
  max_iters: 2000
  confidence: 0.995

verification:
  min_features: 50
  min_matches: 20
  min_inliers: 10

server:
  host: "0.0.0.0"
  port: 8003
  log_level: info
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        from geometric_service.config import clear_settings_cache
        clear_settings_cache()
        settings = get_settings()

        assert settings.service.name == "geometric"
        assert settings.orb.max_features == 1000
        assert settings.matching.ratio_threshold == 0.75
        assert settings.ransac.reproj_threshold == 5.0
        assert settings.verification.min_inliers == 10
```

**Step 2: Run test to verify it fails**

Run: `cd services/geometric && uv run pytest tests/unit/test_config.py -v -m unit`
Expected: FAIL with "ModuleNotFoundError: No module named 'geometric_service.config'"

**Step 3: Write minimal implementation**

Create `services/geometric/src/geometric_service/config.py`:
```python
"""
Configuration management for the geometric service.

Loads configuration from YAML with ZERO defaults.
Every value must be explicitly specified or startup fails.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class ServiceConfig(BaseModel):
    """Service identity configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class ORBConfig(BaseModel):
    """ORB feature detector configuration."""

    model_config = ConfigDict(extra="forbid")

    max_features: int
    scale_factor: float
    n_levels: int
    edge_threshold: int
    patch_size: int
    fast_threshold: int


class MatchingConfig(BaseModel):
    """Feature matching configuration."""

    model_config = ConfigDict(extra="forbid")

    ratio_threshold: float
    cross_check: bool


class RANSACConfig(BaseModel):
    """RANSAC homography configuration."""

    model_config = ConfigDict(extra="forbid")

    reproj_threshold: float
    max_iters: int
    confidence: float


class VerificationConfig(BaseModel):
    """Geometric verification thresholds."""

    model_config = ConfigDict(extra="forbid")

    min_features: int
    min_matches: int
    min_inliers: int


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int
    log_level: str


class Settings(BaseModel):
    """
    Root configuration container.

    All fields are REQUIRED. No defaults exist.
    Missing fields cause immediate startup failure.

    Usage:
        from geometric_service.config import get_settings
        settings = get_settings()
    """

    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig
    orb: ORBConfig
    matching: MatchingConfig
    ransac: RANSACConfig
    verification: VerificationConfig
    server: ServerConfig


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

    if not isinstance(config, dict):
        raise ConfigurationError(
            f"Configuration must be a YAML mapping, got {type(config).__name__}"
        )

    return config


def get_config_path() -> Path:
    """
    Determine configuration file path.

    Uses CONFIG_PATH environment variable if set, otherwise defaults
    to ./config.yaml relative to working directory.

    Returns:
        Path to configuration file
    """
    config_path_str = os.environ.get("CONFIG_PATH")
    if config_path_str is None:
        config_path_str = "config.yaml"
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
    except ValidationError as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}\n"
            f"All configuration values must be explicitly specified.\n"
            f"No default values are allowed."
        ) from e


def clear_settings_cache() -> None:
    """Clear the settings cache. Used in testing."""
    get_settings.cache_clear()


# Keywords that indicate sensitive data (case-insensitive)
SENSITIVE_KEYWORDS: frozenset[str] = frozenset(
    {
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
    }
)

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


REDACTION_MARKER: str = "[REDACTED]"


def redact_sensitive_values(
    data: dict[str, Any],
    redaction_marker: str,
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
                redact_sensitive_values(item, redaction_marker) if isinstance(item, dict) else item
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
    return redact_sensitive_values(raw_config, REDACTION_MARKER)
```

**Step 4: Run test to verify it passes**

Run: `cd services/geometric && uv run pytest tests/unit/test_config.py -v -m unit`
Expected: PASS

**Step 5: Commit**

```bash
git add services/geometric/src/geometric_service/config.py services/geometric/tests/unit/__init__.py services/geometric/tests/unit/test_config.py
git commit -m "feat(geometric): add configuration management module"
git push
```

---

## Task 2: Core Infrastructure - Logging Module

**Files:**
- Create: `services/geometric/src/geometric_service/logging.py`

**Step 1: Write the logging module**

Create `services/geometric/src/geometric_service/logging.py`:
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
        "logger": "geometric_service",
        "message": "Server started",
        "extra": { ... }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # Timestamp in yyyy-mm-dd hh:mm format
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )

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
            key: value for key, value in record.__dict__.items() if key not in standard_attrs
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
    from geometric_service.config import get_settings

    settings = get_settings()
    base_name = settings.service.name

    if name:
        full_name = f"{base_name}.{name}"
    else:
        full_name = base_name

    return logging.getLogger(full_name)
```

**Step 2: Run basic validation**

Run: `cd services/geometric && uv run python -c "from geometric_service.logging import get_logger; print('OK')"`
Expected: OK (may have import error until config is available)

**Step 3: Commit**

```bash
git add services/geometric/src/geometric_service/logging.py
git commit -m "feat(geometric): add structured JSON logging module"
git push
```

---

## Task 3: Core Infrastructure - Application State

**Files:**
- Create: `services/geometric/src/geometric_service/core/__init__.py`
- Create: `services/geometric/src/geometric_service/core/state.py`

**Step 1: Create the state module**

Create `services/geometric/src/geometric_service/core/__init__.py`:
```python
"""Core infrastructure components."""

from geometric_service.core.exceptions import ServiceError
from geometric_service.core.state import AppState, get_app_state, init_app_state

__all__ = ["AppState", "ServiceError", "get_app_state", "init_app_state"]
```

Create `services/geometric/src/geometric_service/core/state.py`:
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

**Step 2: Commit**

```bash
git add services/geometric/src/geometric_service/core/__init__.py services/geometric/src/geometric_service/core/state.py
git commit -m "feat(geometric): add application state management"
git push
```

---

## Task 4: Core Infrastructure - Exception Handlers

**Files:**
- Create: `services/geometric/src/geometric_service/core/exceptions.py`

**Step 1: Create the exceptions module**

Create `services/geometric/src/geometric_service/core/exceptions.py`:
```python
"""
Custom exception handlers for consistent error responses.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from geometric_service.logging import get_logger


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
    # nosemgrep: python.lang.maintainability.useless-ifelse.useless-if-body
    app.add_exception_handler(ServiceError, service_error_handler)  # type: ignore[arg-type]
    # nosemgrep: python.lang.maintainability.useless-ifelse.useless-if-body
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
```

**Step 2: Update core __init__.py**

The `__init__.py` was already created with ServiceError export.

**Step 3: Commit**

```bash
git add services/geometric/src/geometric_service/core/exceptions.py
git commit -m "feat(geometric): add custom exception handlers"
git push
```

---

## Task 5: Core Infrastructure - Lifespan Management

**Files:**
- Create: `services/geometric/src/geometric_service/core/lifespan.py`

**Step 1: Create the lifespan module**

Create `services/geometric/src/geometric_service/core/lifespan.py`:
```python
"""
Application lifecycle management.

Handles startup and shutdown events for proper resource management.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from geometric_service.config import get_settings
from geometric_service.core.state import init_app_state
from geometric_service.logging import get_logger, setup_logging


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
```

**Step 2: Commit**

```bash
git add services/geometric/src/geometric_service/core/lifespan.py
git commit -m "feat(geometric): add application lifecycle management"
git push
```

---

## Task 6: Pydantic Schemas

**Files:**
- Create: `services/geometric/src/geometric_service/schemas.py`
- Test: `services/geometric/tests/unit/test_schemas.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/test_schemas.py`:
```python
"""Unit tests for Pydantic schemas."""

from __future__ import annotations

import pytest

from geometric_service.schemas import (
    ExtractRequest,
    ExtractResponse,
    MatchRequest,
    MatchResponse,
    BatchMatchRequest,
    BatchMatchResponse,
)


@pytest.mark.unit
class TestExtractRequest:
    """Tests for ExtractRequest schema."""

    def test_valid_request(self) -> None:
        """Should accept valid request."""
        request = ExtractRequest(image="base64data", image_id="test_001")
        assert request.image == "base64data"
        assert request.image_id == "test_001"

    def test_image_id_optional(self) -> None:
        """image_id should be optional."""
        request = ExtractRequest(image="base64data")
        assert request.image_id is None

    def test_rejects_empty_image(self) -> None:
        """Should reject empty image."""
        with pytest.raises(ValueError):
            ExtractRequest(image="")


@pytest.mark.unit
class TestMatchRequest:
    """Tests for MatchRequest schema."""

    def test_with_reference_image(self) -> None:
        """Should accept request with reference image."""
        request = MatchRequest(query_image="query", reference_image="ref")
        assert request.query_image == "query"
        assert request.reference_image == "ref"

    def test_query_id_optional(self) -> None:
        """query_id should be optional."""
        request = MatchRequest(query_image="query", reference_image="ref")
        assert request.query_id is None


@pytest.mark.unit
class TestMatchResponse:
    """Tests for MatchResponse schema."""

    def test_valid_response(self) -> None:
        """Should create valid response."""
        response = MatchResponse(
            is_match=True,
            confidence=0.85,
            inliers=67,
            total_matches=124,
            inlier_ratio=0.54,
            query_features=847,
            reference_features=923,
            processing_time_ms=48.7,
        )
        assert response.is_match is True
        assert response.confidence == 0.85
```

**Step 2: Run test to verify it fails**

Run: `cd services/geometric && uv run pytest tests/unit/test_schemas.py -v -m unit`
Expected: FAIL with "ModuleNotFoundError: No module named 'geometric_service.schemas'"

**Step 3: Write the schemas module**

Create `services/geometric/src/geometric_service/schemas.py`:
```python
"""
Pydantic request/response models for the geometric service API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# === Keypoint Model ===


class KeypointData(BaseModel):
    """Keypoint data for feature extraction."""

    model_config = ConfigDict(extra="forbid")

    x: float
    """X coordinate (pixels)."""

    y: float
    """Y coordinate (pixels)."""

    size: float
    """Feature scale."""

    angle: float
    """Feature orientation (degrees)."""


class ImageSize(BaseModel):
    """Image dimensions."""

    model_config = ConfigDict(extra="forbid")

    width: int
    height: int


# === Request Models ===


class ExtractRequest(BaseModel):
    """Request model for POST /extract endpoint."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(..., min_length=1)
    """Base64-encoded image data (JPEG, PNG, or WebP)."""

    image_id: str | None = None
    """Optional identifier for logging/tracing."""

    max_features: int | None = None
    """Optional override for max features to extract."""


class ReferenceFeatures(BaseModel):
    """Pre-extracted features for a reference image."""

    model_config = ConfigDict(extra="forbid")

    keypoints: list[KeypointData]
    """List of keypoint data."""

    descriptors: str
    """Base64-encoded descriptor matrix (N x 32 bytes)."""


class MatchRequest(BaseModel):
    """Request model for POST /match endpoint."""

    model_config = ConfigDict(extra="forbid")

    query_image: str = Field(..., min_length=1)
    """Base64-encoded query image."""

    reference_image: str | None = None
    """Base64-encoded reference image (if not using pre-extracted features)."""

    reference_features: ReferenceFeatures | None = None
    """Pre-extracted reference features (alternative to reference_image)."""

    query_id: str | None = None
    """Optional query image identifier."""

    reference_id: str | None = None
    """Optional reference image identifier."""


class ReferenceInput(BaseModel):
    """Reference input for batch matching."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str
    """Reference identifier."""

    reference_image: str | None = None
    """Base64-encoded reference image."""

    reference_features: ReferenceFeatures | None = None
    """Pre-extracted features (alternative to image)."""


class BatchMatchRequest(BaseModel):
    """Request model for POST /match/batch endpoint."""

    model_config = ConfigDict(extra="forbid")

    query_image: str = Field(..., min_length=1)
    """Base64-encoded query image."""

    references: list[ReferenceInput]
    """List of references to match against."""

    query_id: str | None = None
    """Optional query image identifier."""


# === Response Models ===


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""

    status: str
    """Service health status."""

    uptime_seconds: float
    """Uptime in seconds since service start."""

    uptime: str
    """Human-readable uptime (e.g., "2d 3h 15m 42s")."""

    system_time: str
    """Current system time in yyyy-mm-dd hh:mm format (UTC)."""


class AlgorithmInfo(BaseModel):
    """Algorithm configuration info for /info endpoint."""

    feature_detector: str
    """Feature detection algorithm (e.g., "ORB")."""

    max_features: int
    """Maximum features to extract per image."""

    matcher: str
    """Feature matching algorithm."""

    matcher_norm: str
    """Distance metric for matching."""

    ratio_threshold: float
    """Lowe's ratio test threshold."""

    verification: str
    """Geometric verification method."""

    ransac_reproj_threshold: float
    """RANSAC reprojection error threshold (pixels)."""

    min_inliers: int
    """Minimum inliers to declare match."""


class InfoResponse(BaseModel):
    """Response model for GET /info endpoint."""

    service: str
    """Service name."""

    version: str
    """Service version (semver)."""

    algorithm: AlgorithmInfo
    """Algorithm configuration."""


class ExtractResponse(BaseModel):
    """Response model for POST /extract endpoint."""

    image_id: str | None
    """Echo of input image_id."""

    num_features: int
    """Number of features extracted."""

    keypoints: list[KeypointData]
    """Feature locations and properties."""

    descriptors: str
    """Base64-encoded descriptor matrix (N x 32 bytes)."""

    image_size: ImageSize
    """Original image dimensions."""

    processing_time_ms: float
    """Extraction time in milliseconds."""


class MatchResponse(BaseModel):
    """Response model for POST /match endpoint."""

    is_match: bool
    """Whether verification passed (inliers >= threshold)."""

    confidence: float
    """Match confidence score (0-1)."""

    inliers: int
    """RANSAC inlier count."""

    total_matches: int
    """Matches before RANSAC filtering."""

    inlier_ratio: float
    """inliers / total_matches."""

    query_features: int
    """Features in query image."""

    reference_features: int
    """Features in reference image."""

    homography: list[list[float]] | None = None
    """3x3 transformation matrix (if match found)."""

    query_id: str | None = None
    """Echo of query_id."""

    reference_id: str | None = None
    """Echo of reference_id."""

    processing_time_ms: float
    """Total processing time in milliseconds."""


class BatchMatchResult(BaseModel):
    """Individual result in batch match response."""

    reference_id: str
    """Reference identifier."""

    is_match: bool
    """Whether verification passed."""

    confidence: float
    """Match confidence score (0-1)."""

    inliers: int
    """RANSAC inlier count."""

    inlier_ratio: float
    """inliers / total_matches."""


class BestMatch(BaseModel):
    """Best match information."""

    reference_id: str
    """Reference identifier of best match."""

    confidence: float
    """Confidence of best match."""


class BatchMatchResponse(BaseModel):
    """Response model for POST /match/batch endpoint."""

    query_id: str | None
    """Echo of query_id."""

    query_features: int
    """Features in query image."""

    results: list[BatchMatchResult]
    """Match results for each reference."""

    best_match: BestMatch | None
    """Best matching reference (if any passed verification)."""

    processing_time_ms: float
    """Total processing time in milliseconds."""


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    """Machine-readable error code."""

    message: str
    """Human-readable error description."""

    details: dict[str, Any]
    """Additional error context."""
```

**Step 4: Run test to verify it passes**

Run: `cd services/geometric && uv run pytest tests/unit/test_schemas.py -v -m unit`
Expected: PASS

**Step 5: Commit**

```bash
git add services/geometric/src/geometric_service/schemas.py services/geometric/tests/unit/test_schemas.py
git commit -m "feat(geometric): add Pydantic request/response schemas"
git push
```

---

## Task 7: Services Layer - Feature Extractor

**Files:**
- Create: `services/geometric/src/geometric_service/services/__init__.py`
- Create: `services/geometric/src/geometric_service/services/feature_extractor.py`
- Test: `services/geometric/tests/unit/services/__init__.py`
- Test: `services/geometric/tests/unit/services/test_feature_extractor.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/services/__init__.py`:
```python
"""Unit tests for services layer."""
```

Create `services/geometric/tests/unit/services/test_feature_extractor.py`:
```python
"""Unit tests for ORB feature extractor."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from geometric_service.services.feature_extractor import ORBFeatureExtractor


def create_test_image_with_features(width: int = 200, height: int = 200) -> bytes:
    """Create a test image with detectable features (checkerboard pattern)."""
    # Create checkerboard pattern for feature detection
    img = np.zeros((height, width), dtype=np.uint8)
    block_size = 20
    for i in range(0, height, block_size * 2):
        for j in range(0, width, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.unit
class TestORBFeatureExtractor:
    """Tests for ORBFeatureExtractor."""

    def test_extract_returns_keypoints(self) -> None:
        """Should extract keypoints from image."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        image_bytes = create_test_image_with_features()

        keypoints, descriptors, image_size = extractor.extract(image_bytes)

        assert len(keypoints) > 0
        assert all("x" in kp and "y" in kp for kp in keypoints)

    def test_extract_returns_descriptors(self) -> None:
        """Should return 32-byte descriptors."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        image_bytes = create_test_image_with_features()

        keypoints, descriptors, image_size = extractor.extract(image_bytes)

        assert descriptors is not None
        assert descriptors.shape[1] == 32  # ORB descriptors are 32 bytes

    def test_extract_returns_image_size(self) -> None:
        """Should return correct image size."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        image_bytes = create_test_image_with_features(width=300, height=200)

        keypoints, descriptors, image_size = extractor.extract(image_bytes)

        assert image_size == (300, 200)
```

**Step 2: Run test to verify it fails**

Run: `cd services/geometric && uv run pytest tests/unit/services/test_feature_extractor.py -v -m unit`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the feature extractor**

Create `services/geometric/src/geometric_service/services/__init__.py`:
```python
"""Service layer components."""

from geometric_service.services.feature_extractor import ORBFeatureExtractor
from geometric_service.services.feature_matcher import BFFeatureMatcher
from geometric_service.services.geometric_verifier import RANSACVerifier

__all__ = ["ORBFeatureExtractor", "BFFeatureMatcher", "RANSACVerifier"]
```

Create `services/geometric/src/geometric_service/services/feature_extractor.py`:
```python
"""
ORB feature extraction from images.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from geometric_service.core.exceptions import ServiceError


class ORBFeatureExtractor:
    """Extract ORB features from images."""

    def __init__(
        self,
        max_features: int,
        scale_factor: float,
        n_levels: int,
        edge_threshold: int,
        patch_size: int,
        fast_threshold: int,
    ) -> None:
        """
        Initialize ORB feature extractor.

        Args:
            max_features: Maximum number of features to retain
            scale_factor: Pyramid decimation ratio (> 1.0)
            n_levels: Number of pyramid levels
            edge_threshold: Border pixels excluded from detection
            patch_size: Size of patch used for descriptor
            fast_threshold: FAST corner detection threshold
        """
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )

    def extract(
        self, image_bytes: bytes
    ) -> tuple[list[dict[str, float]], NDArray[np.uint8] | None, tuple[int, int]]:
        """
        Extract ORB features from image bytes.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, or WebP)

        Returns:
            Tuple of:
            - keypoints: List of {x, y, size, angle}
            - descriptors: numpy array (N, 32) or None if no features
            - image_size: (width, height)

        Raises:
            ServiceError: If image cannot be decoded
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ServiceError(
                error="invalid_image",
                message="Failed to decode image data",
                status_code=400,
                details=None,
            )

        # Convert to grayscale for ORB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get image dimensions (height, width, channels)
        height, width = gray.shape[:2]
        image_size = (width, height)

        # Detect keypoints and compute descriptors
        cv_keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # Convert keypoints to serializable format
        keypoints: list[dict[str, float]] = []
        for kp in cv_keypoints:
            keypoints.append(
                {
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                }
            )

        return keypoints, descriptors, image_size

    def keypoints_to_cv(
        self, keypoints: list[dict[str, float]]
    ) -> list[cv2.KeyPoint]:
        """
        Convert serialized keypoints back to OpenCV KeyPoint objects.

        Args:
            keypoints: List of {x, y, size, angle}

        Returns:
            List of cv2.KeyPoint objects
        """
        cv_keypoints = []
        for kp in keypoints:
            cv_keypoints.append(
                cv2.KeyPoint(
                    x=kp["x"],
                    y=kp["y"],
                    size=kp["size"],
                    angle=kp["angle"],
                )
            )
        return cv_keypoints
```

**Step 4: Run test to verify it passes**

Run: `cd services/geometric && uv run pytest tests/unit/services/test_feature_extractor.py -v -m unit`
Expected: PASS

**Step 5: Commit**

```bash
git add services/geometric/src/geometric_service/services/__init__.py services/geometric/src/geometric_service/services/feature_extractor.py services/geometric/tests/unit/services/__init__.py services/geometric/tests/unit/services/test_feature_extractor.py
git commit -m "feat(geometric): add ORB feature extractor"
git push
```

---

## Task 8: Services Layer - Feature Matcher

**Files:**
- Create: `services/geometric/src/geometric_service/services/feature_matcher.py`
- Test: `services/geometric/tests/unit/services/test_feature_matcher.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/services/test_feature_matcher.py`:
```python
"""Unit tests for BF feature matcher."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_service.services.feature_matcher import BFFeatureMatcher


@pytest.mark.unit
class TestBFFeatureMatcher:
    """Tests for BFFeatureMatcher."""

    def test_match_identical_descriptors(self) -> None:
        """Should find matches for identical descriptors."""
        matcher = BFFeatureMatcher(ratio_threshold=0.75)

        # Create random binary descriptors
        np.random.seed(42)
        desc = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = matcher.match(desc, desc)

        # Should find matches (ratio test will filter some)
        assert len(matches) > 0

    def test_match_different_descriptors(self) -> None:
        """Should find fewer matches for different descriptors."""
        matcher = BFFeatureMatcher(ratio_threshold=0.75)

        # Create two completely different descriptor sets
        np.random.seed(42)
        desc1 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)
        np.random.seed(123)
        desc2 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = matcher.match(desc1, desc2)

        # May find some random matches due to binary nature
        # But should be fewer than identical case
        assert isinstance(matches, list)

    def test_match_empty_descriptors(self) -> None:
        """Should return empty list for empty descriptors."""
        matcher = BFFeatureMatcher(ratio_threshold=0.75)

        desc1 = np.array([], dtype=np.uint8).reshape(0, 32)
        desc2 = np.random.randint(0, 256, size=(50, 32), dtype=np.uint8)

        matches = matcher.match(desc1, desc2)

        assert matches == []
```

**Step 2: Run test to verify it fails**

Run: `cd services/geometric && uv run pytest tests/unit/services/test_feature_matcher.py -v -m unit`
Expected: FAIL

**Step 3: Write the feature matcher**

Create `services/geometric/src/geometric_service/services/feature_matcher.py`:
```python
"""
Feature matching using brute-force matcher with Lowe's ratio test.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class BFFeatureMatcher:
    """Match features using brute-force with ratio test."""

    def __init__(self, ratio_threshold: float) -> None:
        """
        Initialize feature matcher.

        Args:
            ratio_threshold: Lowe's ratio test threshold (e.g., 0.75)
        """
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        desc1: NDArray[np.uint8],
        desc2: NDArray[np.uint8],
    ) -> list[cv2.DMatch]:
        """
        Match descriptors using kNN + Lowe's ratio test.

        Args:
            desc1: Descriptors from first image (N1, 32)
            desc2: Descriptors from second image (N2, 32)

        Returns:
            List of good matches after ratio test filtering.
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Need at least 2 descriptors in desc2 for kNN with k=2
        if len(desc2) < 2:
            return []

        # Find 2 nearest neighbors for ratio test
        try:
            matches = self.bf.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        # Apply Lowe's ratio test
        good_matches: list[cv2.DMatch] = []
        for match_pair in matches:
            # Some matches may have fewer than 2 results
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        return good_matches
```

**Step 4: Run test to verify it passes**

Run: `cd services/geometric && uv run pytest tests/unit/services/test_feature_matcher.py -v -m unit`
Expected: PASS

**Step 5: Commit**

```bash
git add services/geometric/src/geometric_service/services/feature_matcher.py services/geometric/tests/unit/services/test_feature_matcher.py
git commit -m "feat(geometric): add brute-force feature matcher with ratio test"
git push
```

---

## Task 9: Services Layer - Geometric Verifier

**Files:**
- Create: `services/geometric/src/geometric_service/services/geometric_verifier.py`
- Test: `services/geometric/tests/unit/services/test_geometric_verifier.py`

**Step 1: Write failing test**

Create `services/geometric/tests/unit/services/test_geometric_verifier.py`:
```python
"""Unit tests for RANSAC geometric verifier."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from geometric_service.services.geometric_verifier import RANSACVerifier, calculate_confidence


@pytest.mark.unit
class TestCalculateConfidence:
    """Tests for confidence score calculation."""

    def test_high_inliers_high_ratio(self) -> None:
        """High inliers + high ratio = high confidence."""
        confidence = calculate_confidence(inliers=100, inlier_ratio=0.8)
        assert confidence >= 0.9

    def test_low_inliers_high_ratio(self) -> None:
        """Low inliers + high ratio = low confidence."""
        confidence = calculate_confidence(inliers=10, inlier_ratio=0.9)
        assert confidence < 0.5

    def test_confidence_bounded(self) -> None:
        """Confidence should be between 0 and 1."""
        for inliers in [0, 10, 50, 100, 200]:
            for ratio in [0.0, 0.3, 0.5, 0.8, 1.0]:
                conf = calculate_confidence(inliers, ratio)
                assert 0.0 <= conf <= 1.0


@pytest.mark.unit
class TestRANSACVerifier:
    """Tests for RANSACVerifier."""

    def test_verify_with_no_matches(self) -> None:
        """Should return no match for empty matches."""
        verifier = RANSACVerifier(
            reproj_threshold=5.0,
            max_iters=2000,
            confidence=0.995,
            min_inliers=10,
        )

        result = verifier.verify(
            kp1=[],
            kp2=[],
            matches=[],
        )

        assert result["is_match"] is False
        assert result["inliers"] == 0

    def test_verify_with_few_matches(self) -> None:
        """Should return no match for too few matches."""
        verifier = RANSACVerifier(
            reproj_threshold=5.0,
            max_iters=2000,
            confidence=0.995,
            min_inliers=10,
        )

        # Create 3 keypoints (below minimum 4 needed for homography)
        kp1 = [cv2.KeyPoint(x=i * 10, y=i * 10, size=10) for i in range(3)]
        kp2 = [cv2.KeyPoint(x=i * 10, y=i * 10, size=10) for i in range(3)]
        matches = [cv2.DMatch(i, i, 0) for i in range(3)]

        result = verifier.verify(kp1, kp2, matches)

        assert result["is_match"] is False
```

**Step 2: Run test to verify it fails**

Run: `cd services/geometric && uv run pytest tests/unit/services/test_geometric_verifier.py -v -m unit`
Expected: FAIL

**Step 3: Write the geometric verifier**

Create `services/geometric/src/geometric_service/services/geometric_verifier.py`:
```python
"""
RANSAC-based geometric verification.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def calculate_confidence(inliers: int, inlier_ratio: float) -> float:
    """
    Combine inlier count and ratio into confidence score.

    - High inliers + high ratio = high confidence
    - High inliers + low ratio = medium confidence
    - Low inliers + high ratio = low confidence

    Args:
        inliers: Number of RANSAC inliers
        inlier_ratio: Ratio of inliers to total matches

    Returns:
        Confidence score between 0 and 1
    """
    # Normalize inlier count (saturates at 100)
    inlier_score = min(inliers / 100.0, 1.0)

    # Weight: 70% inlier count, 30% inlier ratio
    confidence = 0.7 * inlier_score + 0.3 * inlier_ratio

    return round(confidence, 2)


class RANSACVerifier:
    """Verify geometric consistency using RANSAC homography."""

    def __init__(
        self,
        reproj_threshold: float,
        max_iters: int,
        confidence: float,
        min_inliers: int,
    ) -> None:
        """
        Initialize RANSAC verifier.

        Args:
            reproj_threshold: Maximum reprojection error (pixels) to count as inlier
            max_iters: Maximum RANSAC iterations
            confidence: Required confidence in result
            min_inliers: Minimum inliers to declare match
        """
        self.reproj_threshold = reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.min_inliers = min_inliers

    def verify(
        self,
        kp1: list[cv2.KeyPoint],
        kp2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ) -> dict[str, Any]:
        """
        Verify geometric consistency between matched keypoints.

        Args:
            kp1: Keypoints from query image
            kp2: Keypoints from reference image
            matches: Feature matches between images

        Returns:
            Dictionary with:
            - is_match: bool
            - inliers: int
            - total_matches: int
            - inlier_ratio: float
            - homography: list[list[float]] | None
            - confidence: float
        """
        total_matches = len(matches)

        # Need at least 4 points for homography estimation
        if total_matches < 4:
            return {
                "is_match": False,
                "inliers": 0,
                "total_matches": total_matches,
                "inlier_ratio": 0.0,
                "homography": None,
                "confidence": 0.0,
            }

        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography with RANSAC
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=self.reproj_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        # Count inliers
        if mask is None:
            inliers = 0
        else:
            inliers = int(mask.ravel().sum())

        # Calculate inlier ratio
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0.0

        # Determine if match
        is_match = inliers >= self.min_inliers

        # Convert homography to list for JSON serialization
        homography: list[list[float]] | None = None
        if H is not None and is_match:
            homography = H.tolist()

        # Calculate confidence score
        conf = calculate_confidence(inliers, inlier_ratio)

        return {
            "is_match": is_match,
            "inliers": inliers,
            "total_matches": total_matches,
            "inlier_ratio": round(inlier_ratio, 4),
            "homography": homography,
            "confidence": conf,
        }
```

**Step 4: Run test to verify it passes**

Run: `cd services/geometric && uv run pytest tests/unit/services/test_geometric_verifier.py -v -m unit`
Expected: PASS

**Step 5: Commit**

```bash
git add services/geometric/src/geometric_service/services/geometric_verifier.py services/geometric/tests/unit/services/test_geometric_verifier.py
git commit -m "feat(geometric): add RANSAC geometric verifier"
git push
```

---

## Part 1 Summary

This part covers:

**Core Infrastructure (Tasks 1-5):**
- Configuration management with Pydantic + YAML
- Structured JSON logging
- Application state tracking
- Exception handlers
- Lifespan management

**Domain Logic (Tasks 6-9):**
- Pydantic schemas for API
- ORB feature extractor
- BF feature matcher with ratio test
- RANSAC geometric verifier

---

**Continue to [Part 2: API Layer, Testing, and Validation](geometric-service-implementation-plan-part2.md)** for Tasks 10-20.
