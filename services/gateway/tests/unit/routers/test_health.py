"""Tests for health and info endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_all_backends_healthy(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test health check when all backends are healthy."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["backends"]["embeddings"] == "healthy"
        assert data["backends"]["search"] == "healthy"
        assert data["backends"]["geometric"] == "healthy"

    def test_health_without_backend_check(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test health check without checking backends."""
        response = test_client.get("/health?check_backends=false")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["backends"] is None

    def test_health_embeddings_unhealthy(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test health check when embeddings service is unhealthy."""
        mock_app_state.embeddings_client.health_check.return_value = "unavailable"

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["backends"]["embeddings"] == "unavailable"

    def test_health_search_unhealthy(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test health check when search service is unhealthy."""
        mock_app_state.search_client.health_check.return_value = "unavailable"

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["backends"]["search"] == "unavailable"

    def test_health_geometric_unhealthy(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test health check when geometric service is unhealthy (degraded)."""
        mock_app_state.geometric_client.health_check.return_value = "unavailable"

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # Geometric is optional, so status should be degraded, not unhealthy
        assert data["status"] == "degraded"
        assert data["backends"]["geometric"] == "unavailable"


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info endpoint."""

    def test_info_returns_pipeline_config(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test that info endpoint returns pipeline configuration."""
        response = test_client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "gateway"
        assert data["version"] == "0.1.0"
        assert "pipeline" in data
        assert data["pipeline"]["search_k"] == 5
        assert data["pipeline"]["similarity_threshold"] == 0.7
        assert data["pipeline"]["geometric_verification"] is True

    def test_info_returns_backend_info(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test that info endpoint returns backend information."""
        response = test_client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "backends" in data
        assert data["backends"]["embeddings"]["status"] == "healthy"
        assert data["backends"]["search"]["status"] == "healthy"
        assert data["backends"]["geometric"]["status"] == "healthy"
