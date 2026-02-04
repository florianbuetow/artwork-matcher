"""Unit tests for health endpoint."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        """Health endpoint returns 200 OK."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")

            assert response.status_code == 200

    def test_health_returns_healthy_status(self) -> None:
        """Health endpoint returns healthy status."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert data["status"] == "healthy"

    def test_health_system_time_format(self) -> None:
        """System time is in yyyy-mm-dd hh:mm format."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "geometric"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.uptime_seconds = 123.45
            state.uptime_formatted = "2m 3s"
            mock_state.return_value = state

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
            assert re.match(pattern, data["system_time"])
