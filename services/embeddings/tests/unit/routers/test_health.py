"""
Unit tests for health endpoint.

Tests the /health endpoint with mocked dependencies.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from embeddings_service.app import create_app


@pytest.mark.unit
@pytest.mark.usefixtures("mock_settings")
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, mock_app_state: MagicMock) -> None:
        """Health endpoint returns 200 OK."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")

            assert response.status_code == 200

    def test_health_returns_healthy_status(self, mock_app_state: MagicMock) -> None:
        """Health endpoint returns healthy status."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert data["status"] == "healthy"

    def test_health_includes_uptime_seconds(self, mock_app_state: MagicMock) -> None:
        """Health endpoint includes uptime_seconds."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert data["uptime_seconds"] == 123.45

    def test_health_includes_formatted_uptime(self, mock_app_state: MagicMock) -> None:
        """Health endpoint includes formatted uptime."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert data["uptime"] == "2m 3s"

    def test_health_includes_system_time(self, mock_app_state: MagicMock) -> None:
        """Health endpoint includes system_time."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert "system_time" in data

    def test_health_system_time_format(self, mock_app_state: MagicMock) -> None:
        """System time is in yyyy-mm-dd hh:mm format."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            # Format: yyyy-mm-dd hh:mm
            pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
            assert re.match(pattern, data["system_time"]), (
                f"Expected format yyyy-mm-dd hh:mm, got {data['system_time']}"
            )

    def test_health_returns_unhealthy_when_model_not_loaded(
        self,
        mock_app_state: MagicMock,
    ) -> None:
        """Health endpoint returns unhealthy when model state is missing."""
        mock_app_state.model = None

        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.health.get_app_state") as mock_state,
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")
            data = response.json()

            assert response.status_code == 200
            assert data["status"] == "unhealthy"
