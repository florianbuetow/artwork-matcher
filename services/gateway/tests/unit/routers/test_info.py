"""
Unit tests for info endpoint.

Tests the /info endpoint with mocked dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_info_returns_200(self, test_client: TestClient) -> None:
        """Info endpoint returns 200 OK."""
        response = test_client.get("/info")

        assert response.status_code == 200

    def test_info_includes_service_name(self, test_client: TestClient) -> None:
        """Info endpoint includes service name."""
        response = test_client.get("/info")
        data = response.json()

        assert data["service"] == "gateway"

    def test_info_includes_version(self, test_client: TestClient) -> None:
        """Info endpoint includes version."""
        response = test_client.get("/info")
        data = response.json()

        assert data["version"] == "0.1.0"

    def test_info_includes_pipeline_config(self, test_client: TestClient) -> None:
        """Info endpoint includes pipeline configuration."""
        response = test_client.get("/info")
        data = response.json()

        assert "pipeline" in data
        pipeline = data["pipeline"]
        assert "search_k" in pipeline
        assert "similarity_threshold" in pipeline
        assert "geometric_verification" in pipeline
        assert "confidence_threshold" in pipeline

    def test_info_pipeline_search_k(self, test_client: TestClient) -> None:
        """Pipeline search_k is configured correctly."""
        response = test_client.get("/info")
        data = response.json()

        assert data["pipeline"]["search_k"] == 5

    def test_info_pipeline_similarity_threshold(self, test_client: TestClient) -> None:
        """Pipeline similarity_threshold is configured correctly."""
        response = test_client.get("/info")
        data = response.json()

        assert data["pipeline"]["similarity_threshold"] == 0.7

    def test_info_pipeline_geometric_verification(self, test_client: TestClient) -> None:
        """Pipeline geometric_verification is enabled."""
        response = test_client.get("/info")
        data = response.json()

        assert data["pipeline"]["geometric_verification"] is True

    def test_info_pipeline_confidence_threshold(self, test_client: TestClient) -> None:
        """Pipeline confidence_threshold is configured correctly."""
        response = test_client.get("/info")
        data = response.json()

        assert data["pipeline"]["confidence_threshold"] == 0.6

    def test_info_includes_backends(self, test_client: TestClient) -> None:
        """Info endpoint includes backend information."""
        response = test_client.get("/info")
        data = response.json()

        assert "backends" in data
        backends = data["backends"]
        assert "embeddings" in backends
        assert "search" in backends
        assert "geometric" in backends

    def test_info_backend_urls(self, test_client: TestClient) -> None:
        """Backend URLs are included in response."""
        response = test_client.get("/info")
        data = response.json()

        assert data["backends"]["embeddings"]["url"] == "http://localhost:8001"
        assert data["backends"]["search"]["url"] == "http://localhost:8002"
        assert data["backends"]["geometric"]["url"] == "http://localhost:8003"

    def test_info_backend_status(self, test_client: TestClient) -> None:
        """Backend status is included in response."""
        response = test_client.get("/info")
        data = response.json()

        # Mocked backends return "healthy"
        assert data["backends"]["embeddings"]["status"] == "healthy"
        assert data["backends"]["search"]["status"] == "healthy"
        assert data["backends"]["geometric"]["status"] == "healthy"

    def test_info_backend_info(self, test_client: TestClient) -> None:
        """Backend info is included in response (when available)."""
        response = test_client.get("/info")
        data = response.json()

        # Mocked backends return detailed info
        embeddings_info = data["backends"]["embeddings"]["info"]
        assert embeddings_info is not None
        assert embeddings_info["service"] == "embeddings"

    def test_info_response_structure(self, test_client: TestClient) -> None:
        """Info response has correct structure."""
        response = test_client.get("/info")
        data = response.json()

        # Check top-level fields
        assert "service" in data
        assert "version" in data
        assert "pipeline" in data
        assert "backends" in data

        # Check pipeline structure
        pipeline = data["pipeline"]
        assert isinstance(pipeline["search_k"], int)
        assert isinstance(pipeline["similarity_threshold"], float)
        assert isinstance(pipeline["geometric_verification"], bool)
        assert isinstance(pipeline["confidence_threshold"], float)

        # Check backends structure
        for backend_name in ["embeddings", "search", "geometric"]:
            backend = data["backends"][backend_name]
            assert "url" in backend
            assert "status" in backend
