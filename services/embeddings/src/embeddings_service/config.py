"""
Configuration management for the embeddings service.

Loads configuration from YAML with ZERO defaults.
Every value must be explicitly specified or startup fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict
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
    "ConfigurationError",
    "LoggingConfig",
    "ModelConfig",
    "PreprocessingConfig",
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


class ModelConfig(BaseModel):
    """DINOv2 model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    device: Literal["auto", "cpu", "cuda", "mps"]
    embedding_dimension: int
    cache_dir: str
    revision: str


class PreprocessingConfig(BaseModel):
    """Image preprocessing configuration."""

    model_config = ConfigDict(extra="forbid")

    image_size: int
    normalize: bool


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
        from embeddings_service.config import get_settings
        settings = get_settings()
    """

    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    server: ServerConfig
    logging: LoggingConfig


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
