"""Tests for health endpoint."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def client(mock_settings: MagicMock, mock_app_state: MagicMock) -> Iterator[TestClient]:
    """Create test client with mocked dependencies."""
    # Patch get_settings and get_app_state in all modules where they're used
    with (
        patch("search_service.config.get_settings", return_value=mock_settings),
        patch("search_service.core.state.get_app_state", return_value=mock_app_state),
        patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
        patch("search_service.routers.health.get_app_state", return_value=mock_app_state),
    ):
        from search_service.app import create_app

        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_healthy(self, client: TestClient) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "uptime" in data
        assert "system_time" in data

    def test_health_system_time_format(self, client: TestClient) -> None:
        """System time is in correct format."""
        response = client.get("/health")

        data = response.json()
        # Format: yyyy-mm-dd hh:mm
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
        assert re.match(pattern, data["system_time"])

    def test_health_returns_uptime(self, client: TestClient, mock_app_state: MagicMock) -> None:
        """Health endpoint returns uptime information."""
        response = client.get("/health")

        data = response.json()
        assert data["uptime_seconds"] == mock_app_state.uptime_seconds
        assert data["uptime"] == mock_app_state.uptime_formatted
