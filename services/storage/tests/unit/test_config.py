"""
Unit tests for configuration management.

Tests YAML loading, Settings validation, and sensitive data redaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from storage_service.config import (
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


VALID_CONFIG = {
    "service": {"name": "storage", "version": "0.1.0"},
    "storage": {"path": "./data/objects", "content_type": "application/octet-stream"},
    "server": {"host": "0.0.0.0", "port": 8004, "log_level": "info"},
    "logging": {"level": "INFO", "format": "json"},
}

VALID_CONFIG_YAML = """
service:
  name: storage
  version: 0.1.0

storage:
  path: "./data/objects"
  content_type: "application/octet-stream"

server:
  host: "0.0.0.0"
  port: 8004
  log_level: info

logging:
  level: "INFO"
  format: "json"
"""


@pytest.mark.unit
class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Valid YAML file loads successfully."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(VALID_CONFIG_YAML)

        result = load_yaml_config(config_file)

        assert result["service"]["name"] == "storage"
        assert result["server"]["port"] == 8004

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
        settings = Settings(**VALID_CONFIG)

        assert settings.service.name == "storage"
        assert settings.server.port == 8004
        assert settings.storage.path == "./data/objects"
        assert settings.storage.content_type == "application/octet-stream"

    def test_missing_required_field_raises_error(self) -> None:
        """Missing required field raises ValidationError."""
        config = {**VALID_CONFIG, "service": {"name": "storage"}}

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "version" in str(exc_info.value)

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError (extra='forbid')."""
        config = {
            **VALID_CONFIG,
            "server": {**VALID_CONFIG["server"], "unknown": "value"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "extra" in str(exc_info.value).lower()

    def test_invalid_port_type_raises_error(self) -> None:
        """Invalid port value raises ValidationError."""
        config = {
            **VALID_CONFIG,
            "server": {**VALID_CONFIG["server"], "port": "not_a_number"},
        }

        with pytest.raises(ValidationError):
            Settings(**config)

    def test_missing_section_raises_error(self) -> None:
        """Missing entire section raises ValidationError."""
        config = {k: v for k, v in VALID_CONFIG.items() if k != "storage"}

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "storage" in str(exc_info.value)


@pytest.mark.unit
class TestGetSettings:
    """Tests for get_settings function."""

    def test_loads_valid_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Should load valid configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(VALID_CONFIG_YAML)
        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        clear_settings_cache()
        settings = get_settings()

        assert settings.service.name == "storage"
        assert settings.storage.path == "./data/objects"
        assert settings.storage.content_type == "application/octet-stream"


@pytest.mark.unit
class TestSensitiveKeyDetection:
    """Tests for sensitive key detection."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("api_key", True),
            ("password", True),
            ("secret", True),
            ("token", True),
            ("normal_field", False),
            ("host", False),
            ("port", False),
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
