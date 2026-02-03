"""
Integration tests for gateway API endpoints.

These tests use the real gateway application with HTTP-level mocking
for backend services via respx. They verify the complete request flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import respx
    from fastapi.testclient import TestClient


# =============================================================================
# Health Endpoint Integration Tests
# =============================================================================


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for GET /health."""

    def test_health_returns_200(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Health endpoint returns healthy status when all backends healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_backend_status(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Health endpoint includes backend status."""
        response = client.get("/health")
        data = response.json()
        assert "backends" in data
        assert data["backends"]["embeddings"] == "healthy"
        assert data["backends"]["search"] == "healthy"
        assert data["backends"]["geometric"] == "healthy"

    def test_health_without_backend_check(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Health endpoint can skip backend checks."""
        response = client.get("/health?check_backends=false")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["backends"] is None


# =============================================================================
# Info Endpoint Integration Tests
# =============================================================================


@pytest.mark.integration
class TestInfoEndpoint:
    """Integration tests for GET /info."""

    def test_info_returns_200(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_includes_service_name(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Info endpoint includes service name."""
        response = client.get("/info")
        data = response.json()
        assert data["service"] == "gateway"

    def test_info_includes_version(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Info endpoint includes version."""
        response = client.get("/info")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_info_includes_pipeline_config(
        self,
        client: TestClient,
        mock_backend_healthy: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Info endpoint includes pipeline configuration."""
        response = client.get("/info")
        data = response.json()
        assert "pipeline" in data
        pipeline = data["pipeline"]
        assert "search_k" in pipeline
        assert "similarity_threshold" in pipeline
        assert "geometric_verification" in pipeline


# =============================================================================
# Identify Endpoint Integration Tests - Success Cases
# =============================================================================


@pytest.mark.integration
class TestIdentifySuccess:
    """Integration tests for successful identification."""

    def test_identify_returns_200(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,  # noqa: ARG002
        sample_image_base64: str,
    ) -> None:
        """Identify endpoint returns 200 OK for valid image."""
        response = client.post(
            "/identify",
            json={"image": sample_image_base64},
        )
        assert response.status_code == 200

    def test_identify_returns_match(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,  # noqa: ARG002
        sample_image_base64: str,
    ) -> None:
        """Identify endpoint returns match details."""
        response = client.post(
            "/identify",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert data["success"] is True
        assert data["match"] is not None
        assert data["match"]["object_id"] == "artwork_001"
        assert "confidence" in data["match"]

    def test_identify_returns_timing(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,  # noqa: ARG002
        sample_image_base64: str,
    ) -> None:
        """Identify endpoint returns timing information."""
        response = client.post(
            "/identify",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert "timing" in data
        timing = data["timing"]
        assert "embedding_ms" in timing
        assert "search_ms" in timing
        assert "total_ms" in timing

    def test_identify_no_match(
        self,
        client: TestClient,
        mock_identify_no_match: respx.MockRouter,  # noqa: ARG002
        sample_image_base64: str,
    ) -> None:
        """Identify endpoint handles no match gracefully."""
        response = client.post(
            "/identify",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert data["success"] is True
        assert data["match"] is None
        assert "message" in data


# =============================================================================
# Identify Endpoint Integration Tests - Error Handling
# =============================================================================


@pytest.mark.integration
class TestIdentifyErrors:
    """Integration tests for error handling."""

    def test_identify_missing_image(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Identify endpoint requires image field."""
        response = client.post(
            "/identify",
            json={},
        )
        assert response.status_code == 422  # Validation error

    def test_identify_empty_image(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Identify endpoint rejects empty image."""
        response = client.post(
            "/identify",
            json={"image": ""},
        )
        assert response.status_code == 422  # Validation error


# =============================================================================
# Error Propagation Tests
# =============================================================================


@pytest.mark.integration
class TestErrorPropagation:
    """Integration tests for backend error propagation."""

    def test_health_shows_unhealthy_backend(
        self,
        client: TestClient,
        mock_embeddings_unavailable: respx.MockRouter,  # noqa: ARG002
    ) -> None:
        """Health endpoint shows unhealthy when backend is unavailable."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["backends"]["embeddings"] == "unavailable"
