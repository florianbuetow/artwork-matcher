"""Unit tests for info endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_info_returns_200(self) -> None:
        """Info endpoint returns 200 OK."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")

            assert response.status_code == 200

    def test_info_returns_algorithm_config(self) -> None:
        """Info endpoint returns algorithm configuration."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            settings.orb.max_features = 1000
            settings.matching.ratio_threshold = 0.75
            settings.ransac.reproj_threshold = 5.0
            settings.verification.min_inliers = 10
            mock_settings.return_value = settings

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["service"] == "geometric"
            assert data["algorithm"]["feature_detector"] == "ORB"
            assert data["algorithm"]["max_features"] == 1000
