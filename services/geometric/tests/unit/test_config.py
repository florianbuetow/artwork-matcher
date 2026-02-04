"""
Unit tests for configuration management.

Tests YAML loading, Settings validation, and sensitive data redaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from geometric_service.config import (
    REDACTION_MARKER,
    ConfigurationError,
    Settings,
    clear_settings_cache,
    get_settings,
    is_sensitive_key,
    load_yaml_config,
    redact_sensitive_values,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Valid YAML file loads successfully."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
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
  host: localhost
  port: 8003
  log_level: info
"""
        )

        result = load_yaml_config(config_file)

        assert result["service"]["name"] == "geometric"
        assert result["server"]["port"] == 8003

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Missing config file raises ConfigurationError."""
        missing_file = tmp_path / "does_not_exist.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config(missing_file)

        assert "not found" in str(exc_info.value)

    def test_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Empty config file raises ConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config(config_file)

        assert "empty" in str(exc_info.value)

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Malformed YAML raises ConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError):
            load_yaml_config(config_file)

    def test_non_dict_yaml_raises_error(self, tmp_path: Path) -> None:
        """YAML that is not a mapping raises ConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config(config_file)

        assert "mapping" in str(exc_info.value)


@pytest.mark.unit
class TestSettings:
    """Tests for Settings validation."""

    def test_valid_config_creates_settings(self) -> None:
        """Valid configuration creates Settings instance."""
        config = {
            "service": {"name": "geometric", "version": "0.1.0"},
            "orb": {
                "max_features": 1000,
                "scale_factor": 1.2,
                "n_levels": 8,
                "edge_threshold": 31,
                "patch_size": 31,
                "fast_threshold": 20,
            },
            "matching": {"ratio_threshold": 0.75, "cross_check": False},
            "ransac": {"reproj_threshold": 5.0, "max_iters": 2000, "confidence": 0.995},
            "verification": {"min_features": 50, "min_matches": 20, "min_inliers": 10},
            "server": {"host": "0.0.0.0", "port": 8003, "log_level": "info"},
        }

        settings = Settings(**config)

        assert settings.service.name == "geometric"
        assert settings.server.port == 8003
        assert settings.orb.max_features == 1000

    def test_missing_required_field_raises_error(self) -> None:
        """Missing required field raises ValidationError."""
        config = {
            "service": {"name": "geometric"},  # Missing 'version'
            "orb": {
                "max_features": 1000,
                "scale_factor": 1.2,
                "n_levels": 8,
                "edge_threshold": 31,
                "patch_size": 31,
                "fast_threshold": 20,
            },
            "matching": {"ratio_threshold": 0.75, "cross_check": False},
            "ransac": {"reproj_threshold": 5.0, "max_iters": 2000, "confidence": 0.995},
            "verification": {"min_features": 50, "min_matches": 20, "min_inliers": 10},
            "server": {"host": "0.0.0.0", "port": 8003, "log_level": "info"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "version" in str(exc_info.value)

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError (extra='forbid')."""
        config = {
            "service": {"name": "geometric", "version": "0.1.0"},
            "orb": {
                "max_features": 1000,
                "scale_factor": 1.2,
                "n_levels": 8,
                "edge_threshold": 31,
                "patch_size": 31,
                "fast_threshold": 20,
            },
            "matching": {"ratio_threshold": 0.75, "cross_check": False},
            "ransac": {"reproj_threshold": 5.0, "max_iters": 2000, "confidence": 0.995},
            "verification": {"min_features": 50, "min_matches": 20, "min_inliers": 10},
            "server": {"host": "0.0.0.0", "port": 8003, "log_level": "info", "unknown": "value"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "extra" in str(exc_info.value).lower()

    def test_invalid_port_type_raises_error(self) -> None:
        """Invalid port value raises ValidationError."""
        config = {
            "service": {"name": "geometric", "version": "0.1.0"},
            "orb": {
                "max_features": 1000,
                "scale_factor": 1.2,
                "n_levels": 8,
                "edge_threshold": 31,
                "patch_size": 31,
                "fast_threshold": 20,
            },
            "matching": {"ratio_threshold": 0.75, "cross_check": False},
            "ransac": {"reproj_threshold": 5.0, "max_iters": 2000, "confidence": 0.995},
            "verification": {"min_features": 50, "min_matches": 20, "min_inliers": 10},
            "server": {"host": "0.0.0.0", "port": "not_a_number", "log_level": "info"},
        }

        with pytest.raises(ValidationError):
            Settings(**config)

    def test_missing_section_raises_error(self) -> None:
        """Missing entire section raises ValidationError."""
        config = {
            "service": {"name": "geometric", "version": "0.1.0"},
            # Missing 'orb' section
            "matching": {"ratio_threshold": 0.75, "cross_check": False},
            "ransac": {"reproj_threshold": 5.0, "max_iters": 2000, "confidence": 0.995},
            "verification": {"min_features": 50, "min_matches": 20, "min_inliers": 10},
            "server": {"host": "0.0.0.0", "port": 8003, "log_level": "info"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "orb" in str(exc_info.value)


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

        clear_settings_cache()
        settings = get_settings()

        assert settings.service.name == "geometric"
        assert settings.orb.max_features == 1000
        assert settings.matching.ratio_threshold == 0.75
        assert settings.ransac.reproj_threshold == 5.0
        assert settings.verification.min_inliers == 10


@pytest.mark.unit
class TestSensitiveKeyDetection:
    """Tests for sensitive key detection."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("api_key", True),
            ("apikey", True),
            ("API_KEY", True),
            ("password", True),
            ("db_password", True),
            ("secret", True),
            ("client_secret", True),
            ("token", True),
            ("auth_token", True),
            ("credential", True),
            ("private_key", True),
            ("bearer_token", True),
            ("normal_field", False),
            ("username", False),
            ("host", False),
            ("port", False),
            ("name", False),
            ("version", False),
        ],
    )
    def test_is_sensitive_key(self, key: str, expected: bool) -> None:
        """Correctly identifies sensitive keys."""
        assert is_sensitive_key(key) == expected


@pytest.mark.unit
class TestSensitiveDataRedaction:
    """Tests for sensitive value redaction."""

    def test_redact_sensitive_values_flat(self) -> None:
        """Redacts sensitive values in flat dict."""
        data = {
            "host": "localhost",
            "api_key": "secret123",
            "port": 8000,
        }

        result = redact_sensitive_values(data, REDACTION_MARKER)

        assert result["host"] == "localhost"
        assert result["api_key"] == REDACTION_MARKER
        assert result["port"] == 8000

    def test_redact_sensitive_values_nested(self) -> None:
        """Redacts sensitive values in nested dict."""
        data = {
            "database": {
                "host": "localhost",
                "password": "secret123",
            },
            "api": {
                "key": "abc123",
            },
        }

        result = redact_sensitive_values(data, REDACTION_MARKER)

        assert result["database"]["host"] == "localhost"
        assert result["database"]["password"] == REDACTION_MARKER
        assert result["api"]["key"] == REDACTION_MARKER

    def test_redact_preserves_list_structure(self) -> None:
        """Redaction preserves list structure."""
        data = {
            "list_field": [1, 2, 3],
            "nested": {"deep": {"value": "keep"}},
        }

        result = redact_sensitive_values(data, REDACTION_MARKER)

        assert result["list_field"] == [1, 2, 3]
        assert result["nested"]["deep"]["value"] == "keep"

    def test_redact_handles_list_of_dicts(self) -> None:
        """Redaction handles lists containing dicts."""
        data = {
            "items": [
                {"name": "item1", "api_key": "secret1"},
                {"name": "item2", "api_key": "secret2"},
            ],
        }

        result = redact_sensitive_values(data, REDACTION_MARKER)

        assert result["items"][0]["name"] == "item1"
        assert result["items"][0]["api_key"] == REDACTION_MARKER
        assert result["items"][1]["api_key"] == REDACTION_MARKER

    def test_redact_uses_custom_marker(self) -> None:
        """Redaction uses provided marker string."""
        data = {"password": "secret"}
        custom_marker = "***HIDDEN***"

        result = redact_sensitive_values(data, custom_marker)

        assert result["password"] == custom_marker
