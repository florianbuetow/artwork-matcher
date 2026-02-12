"""
Configuration management for the geometric service.

Loads configuration from YAML with ZERO defaults.
Every value must be explicitly specified or startup fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    "MatchingConfig",
    "ORBConfig",
    "RANSACConfig",
    "ServerConfig",
    "ServiceConfig",
    "Settings",
    "VerificationConfig",
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
    """RANSAC geometric verification configuration."""

    model_config = ConfigDict(extra="forbid")

    reproj_threshold: float
    max_iters: int
    confidence: float


class VerificationConfig(BaseModel):
    """Verification thresholds configuration."""

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
