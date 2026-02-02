"""
Configuration management for the gateway service.

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


class BackendsConfig(BaseModel):
    """Backend service URLs and timeouts."""

    model_config = ConfigDict(extra="forbid")

    embeddings_url: str
    search_url: str
    geometric_url: str
    timeout_seconds: float


class PipelineConfig(BaseModel):
    """Identification pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    search_k: int
    similarity_threshold: float
    geometric_verification: bool
    confidence_threshold: float


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int
    log_level: str
    cors_origins: list[str]


class DataConfig(BaseModel):
    """Data paths configuration."""

    model_config = ConfigDict(extra="forbid")

    objects_path: str
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
    server: ServerConfig
    data: DataConfig


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

    # Add service identity if not present in YAML
    if "service" not in yaml_config:
        yaml_config["service"] = {"name": "gateway", "version": "0.1.0"}

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
