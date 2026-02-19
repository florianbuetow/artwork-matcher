"""
Configuration management for the gateway service.

Loads configuration from YAML with ZERO defaults.
Every value must be explicitly specified or startup fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from service_commons.config import (
    REDACTION_MARKER,
    SENSITIVE_KEYWORDS,
    ConfigurationError,
    create_settings_loader,
    get_safe_model_config,
    is_sensitive_key,
    load_yaml_config,
    redact_sensitive_values,
)
from service_commons.config import (
    get_config_path as resolve_config_path,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "REDACTION_MARKER",
    "SENSITIVE_KEYWORDS",
    "BackendsConfig",
    "CircuitBreakerConfig",
    "ConfigurationError",
    "DataConfig",
    "PipelineConfig",
    "RetryConfig",
    "ScoringConfig",
    "ServerConfig",
    "ServiceConfig",
    "Settings",
    "clear_settings_cache",
    "get_config_path",
    "get_safe_config",
    "get_settings",
    "is_sensitive_key",
    "load_yaml_config",
    "redact_sensitive_values",
]


class ServiceConfig(BaseModel):
    """Service identity configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class RetryConfig(BaseModel):
    """Retry behavior for backend requests."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(..., ge=1)
    initial_backoff_seconds: float = Field(..., gt=0)
    max_backoff_seconds: float = Field(..., gt=0)
    jitter_seconds: float = Field(..., ge=0)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker thresholds for backend requests."""

    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = Field(..., ge=1)
    recovery_timeout_seconds: float = Field(..., gt=0)


class BackendsConfig(BaseModel):
    """Backend service URLs, timeout, retry, and circuit breaker settings."""

    model_config = ConfigDict(extra="forbid")

    embeddings_url: str
    search_url: str
    geometric_url: str
    storage_url: str
    timeout_seconds: float = Field(..., gt=0)
    retry: RetryConfig
    circuit_breaker: CircuitBreakerConfig


class PipelineConfig(BaseModel):
    """Identification pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    search_k: int = Field(..., ge=1)
    similarity_threshold: float = Field(..., ge=0, le=1)
    geometric_verification: bool
    confidence_threshold: float = Field(..., ge=0, le=1)


class ScoringConfig(BaseModel):
    """Confidence scoring weights and thresholds."""

    model_config = ConfigDict(extra="forbid")

    geometric_score_threshold: float = Field(..., ge=0, le=1)
    geometric_high_similarity_weight: float = Field(..., ge=0, le=1)
    geometric_high_score_weight: float = Field(..., ge=0, le=1)
    geometric_low_similarity_weight: float = Field(..., ge=0, le=1)
    geometric_low_score_weight: float = Field(..., ge=0, le=1)
    geometric_missing_penalty: float = Field(..., ge=0, le=1)
    embedding_only_penalty: float = Field(..., ge=0, le=1)


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = Field(..., ge=1, le=65535)
    log_level: str
    cors_origins: list[str]


class DataConfig(BaseModel):
    """Data paths configuration."""

    model_config = ConfigDict(extra="forbid")

    labels_path: str


class Settings(BaseModel):
    """
    Root configuration container.

    All fields are REQUIRED. No defaults exist.
    Missing fields cause immediate startup failure.

    Usage:
        from gateway.config import get_settings
        settings = get_settings()
    """

    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig
    backends: BackendsConfig
    pipeline: PipelineConfig
    scoring: ScoringConfig
    server: ServerConfig
    data: DataConfig


def get_config_path() -> Path:
    """
    Determine configuration file path.

    Uses CONFIG_PATH environment variable if set, otherwise defaults
    to ./config.yaml relative to working directory.

    Returns:
        Path to configuration file
    """
    return resolve_config_path(
        env_var_name="CONFIG_PATH",
        default_filename="config.yaml",
    )


get_settings, clear_settings_cache = create_settings_loader(Settings, get_config_path)  # nosemgrep


def get_safe_config() -> dict[str, Any]:
    """
    Get configuration with sensitive values redacted.

    Returns:
        Configuration dictionary safe for logging/API exposure
    """
    return get_safe_model_config(get_settings(), REDACTION_MARKER)
