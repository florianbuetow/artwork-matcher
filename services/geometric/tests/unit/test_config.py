"""Unit tests for configuration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from geometric_service.config import (
    ConfigurationError,
    clear_settings_cache,
    get_settings,
    load_yaml_config,
)

if TYPE_CHECKING:
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

        clear_settings_cache()
        settings = get_settings()

        assert settings.service.name == "geometric"
        assert settings.orb.max_features == 1000
        assert settings.matching.ratio_threshold == 0.75
        assert settings.ransac.reproj_threshold == 5.0
        assert settings.verification.min_inliers == 10
