"""Tests for configuration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from pydantic import ValidationError

from search_service.config import (
    Settings,
    clear_settings_cache,
    is_sensitive_key,
    load_yaml_config,
    redact_sensitive_values,
)


@pytest.fixture(autouse=True)
def reset_settings() -> None:
    """Clear settings cache before each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_valid_config_loads(self, tmp_path: Path) -> None:
        """Valid YAML produces valid dictionary."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
service:
  name: search
  version: "0.1.0"
faiss:
  embedding_dimension: 768
  index_type: flat
  metric: inner_product
index:
  path: /data/index/faiss.index
  metadata_path: /data/index/metadata.json
  auto_load: true
search:
  default_k: 5
  max_k: 100
  default_threshold: 0.0
server:
  host: "0.0.0.0"
  port: 8002
logging:
  level: INFO
  format: json
""")

        config = load_yaml_config(config_file)

        assert config["service"]["name"] == "search"
        assert config["faiss"]["embedding_dimension"] == 768

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Missing config file raises ConfigurationError."""
        from search_service.config import ConfigurationError

        missing_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_yaml_config(missing_file)

    def test_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Empty config file raises ConfigurationError."""
        from search_service.config import ConfigurationError

        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ConfigurationError, match="Configuration file is empty"):
            load_yaml_config(empty_file)


class TestSettings:
    """Tests for Settings validation."""

    def test_valid_settings(self) -> None:
        """Valid config dictionary creates valid Settings."""
        config = {
            "service": {"name": "search", "version": "0.1.0"},
            "faiss": {
                "embedding_dimension": 768,
                "index_type": "flat",
                "metric": "inner_product",
            },
            "index": {
                "path": "/data/index/faiss.index",
                "metadata_path": "/data/index/metadata.json",
                "auto_load": True,
            },
            "search": {"default_k": 5, "max_k": 100, "default_threshold": 0.0},
            "server": {"host": "0.0.0.0", "port": 8002},
            "logging": {"level": "INFO", "format": "json"},
        }

        settings = Settings(**config)

        assert settings.service.name == "search"
        assert settings.faiss.embedding_dimension == 768
        assert settings.index.auto_load is True

    def test_missing_required_field_rejected(self) -> None:
        """Missing required field raises ValidationError."""
        config = {
            "service": {"name": "search"},  # Missing version
            "faiss": {
                "embedding_dimension": 768,
                "index_type": "flat",
                "metric": "inner_product",
            },
            "index": {
                "path": "/data/index/faiss.index",
                "metadata_path": "/data/index/metadata.json",
                "auto_load": True,
            },
            "search": {"default_k": 5, "max_k": 100, "default_threshold": 0.0},
            "server": {"host": "0.0.0.0", "port": 8002},
            "logging": {"level": "INFO", "format": "json"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "version" in str(exc_info.value)

    def test_extra_keys_rejected(self) -> None:
        """Unknown configuration keys are rejected."""
        config = {
            "service": {"name": "search", "version": "0.1.0", "unknown_key": "bad"},
            "faiss": {
                "embedding_dimension": 768,
                "index_type": "flat",
                "metric": "inner_product",
            },
            "index": {
                "path": "/data/index/faiss.index",
                "metadata_path": "/data/index/metadata.json",
                "auto_load": True,
            },
            "search": {"default_k": 5, "max_k": 100, "default_threshold": 0.0},
            "server": {"host": "0.0.0.0", "port": 8002},
            "logging": {"level": "INFO", "format": "json"},
        }

        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)

        assert "extra" in str(exc_info.value).lower()


class TestSensitiveKeyDetection:
    """Tests for sensitive key detection and redaction."""

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("api_key", True),
            ("secret", True),
            ("password", True),
            ("token", True),
            ("API_KEY", True),
            ("my_api_key_value", True),
            ("name", False),
            ("host", False),
            ("port", False),
            ("embedding_dimension", False),
        ],
    )
    def test_is_sensitive_key(self, key: str, expected: bool) -> None:
        """Sensitive keywords are detected correctly."""
        assert is_sensitive_key(key) == expected

    def test_redact_sensitive_values(self) -> None:
        """Sensitive values are redacted in nested dicts."""
        data = {
            "service": {"name": "search", "api_key": "secret123"},
            "database": {"host": "localhost", "password": "hunter2"},
        }

        redacted = redact_sensitive_values(data, "[HIDDEN]")

        assert redacted["service"]["name"] == "search"
        assert redacted["service"]["api_key"] == "[HIDDEN]"
        assert redacted["database"]["host"] == "localhost"
        assert redacted["database"]["password"] == "[HIDDEN]"
