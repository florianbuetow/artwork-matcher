"""
Integration tests for all API endpoints.

These tests use the real application with the actual DINOv2 model loaded.
They verify the complete request -> processing -> response cycle.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# =============================================================================
# Health Endpoint Integration Tests
# =============================================================================


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_uptime_seconds(self, client: TestClient) -> None:
        """Health endpoint includes uptime_seconds."""
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_includes_uptime_formatted(self, client: TestClient) -> None:
        """Health endpoint includes formatted uptime."""
        response = client.get("/health")
        data = response.json()
        assert "uptime" in data
        assert isinstance(data["uptime"], str)
        # Should contain at least seconds
        assert "s" in data["uptime"]

    def test_health_includes_system_time(self, client: TestClient) -> None:
        """Health endpoint includes system_time."""
        response = client.get("/health")
        data = response.json()
        assert "system_time" in data

    def test_health_system_time_format(self, client: TestClient) -> None:
        """System time is in yyyy-mm-dd hh:mm format."""
        response = client.get("/health")
        data = response.json()
        # Format: yyyy-mm-dd hh:mm
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
        assert re.match(pattern, data["system_time"]), (
            f"Expected format yyyy-mm-dd hh:mm, got {data['system_time']}"
        )


# =============================================================================
# Info Endpoint Integration Tests
# =============================================================================


@pytest.mark.integration
class TestInfoEndpoint:
    """Integration tests for GET /info."""

    def test_info_returns_200(self, client: TestClient) -> None:
        """Info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_includes_service_name(self, client: TestClient) -> None:
        """Info endpoint includes service name."""
        response = client.get("/info")
        data = response.json()
        assert data["service"] == "embeddings"

    def test_info_includes_version(self, client: TestClient) -> None:
        """Info endpoint includes version."""
        response = client.get("/info")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_info_includes_model_config(self, client: TestClient) -> None:
        """Info endpoint includes model configuration."""
        response = client.get("/info")
        data = response.json()
        assert "model" in data
        model = data["model"]
        assert "name" in model
        assert "embedding_dimension" in model
        assert "device" in model

    def test_info_model_name_is_dinov2(self, client: TestClient) -> None:
        """Model name contains dinov2."""
        response = client.get("/info")
        data = response.json()
        assert "dinov2" in data["model"]["name"].lower()

    def test_info_embedding_dimension_is_768(self, client: TestClient) -> None:
        """Embedding dimension is 768 for dinov2-base."""
        response = client.get("/info")
        data = response.json()
        assert data["model"]["embedding_dimension"] == 768

    def test_info_includes_preprocessing_config(self, client: TestClient) -> None:
        """Info endpoint includes preprocessing configuration."""
        response = client.get("/info")
        data = response.json()
        assert "preprocessing" in data
        preprocessing = data["preprocessing"]
        assert "image_size" in preprocessing
        assert "normalize" in preprocessing

    def test_info_image_size_is_518(self, client: TestClient) -> None:
        """Image size is 518 (DINOv2 native)."""
        response = client.get("/info")
        data = response.json()
        assert data["preprocessing"]["image_size"] == 518

    def test_info_normalize_is_true(self, client: TestClient) -> None:
        """Normalize is enabled."""
        response = client.get("/info")
        data = response.json()
        assert data["preprocessing"]["normalize"] is True


# =============================================================================
# Embed Endpoint Integration Tests - Success Cases
# =============================================================================


@pytest.mark.integration
class TestEmbedSuccess:
    """Integration tests for successful embedding extraction."""

    def test_embed_returns_200(self, client: TestClient, sample_image_base64: str) -> None:
        """Embed endpoint returns 200 OK for valid image."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        assert response.status_code == 200

    def test_embed_returns_embedding(self, client: TestClient, sample_image_base64: str) -> None:
        """Embed endpoint returns embedding array."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) > 0

    def test_embed_returns_correct_dimension(
        self, client: TestClient, sample_image_base64: str
    ) -> None:
        """Embedding has correct dimension (768 for dinov2-base)."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert data["dimension"] == 768
        assert len(data["embedding"]) == 768

    def test_embed_returns_processing_time(
        self, client: TestClient, sample_image_base64: str
    ) -> None:
        """Embed endpoint returns processing time."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] > 0

    def test_embed_echoes_image_id(self, client: TestClient, sample_image_base64: str) -> None:
        """Embed endpoint echoes back image_id."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64, "image_id": "test_001"},
        )
        data = response.json()
        assert data["image_id"] == "test_001"

    def test_embed_image_id_is_optional(self, client: TestClient, sample_image_base64: str) -> None:
        """Embed endpoint works without image_id."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        data = response.json()
        assert data["image_id"] is None

    def test_embed_embedding_is_l2_normalized(
        self, client: TestClient, sample_image_base64: str
    ) -> None:
        """Embedding is L2-normalized (unit length)."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        data = response.json()
        embedding = np.array(data["embedding"])
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"


# =============================================================================
# Embed Endpoint Integration Tests - Image Formats
# =============================================================================


@pytest.mark.integration
class TestEmbedFormats:
    """Integration tests for different image formats."""

    def test_embed_accepts_jpeg(self, client: TestClient, sample_image_base64: str) -> None:
        """Embed endpoint accepts JPEG images."""
        response = client.post(
            "/embed",
            json={"image": sample_image_base64},
        )
        assert response.status_code == 200

    def test_embed_accepts_png(self, client: TestClient, sample_png_base64: str) -> None:
        """Embed endpoint accepts PNG images."""
        response = client.post(
            "/embed",
            json={"image": sample_png_base64},
        )
        assert response.status_code == 200

    def test_embed_accepts_webp(self, client: TestClient, sample_webp_base64: str) -> None:
        """Embed endpoint accepts WebP images."""
        response = client.post(
            "/embed",
            json={"image": sample_webp_base64},
        )
        assert response.status_code == 200

    def test_embed_rejects_bmp(self, client: TestClient, bmp_image_base64: str) -> None:
        """Embed endpoint rejects BMP format."""
        response = client.post(
            "/embed",
            json={"image": bmp_image_base64},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "unsupported_format"


# =============================================================================
# Embed Endpoint Integration Tests - Error Handling
# =============================================================================


@pytest.mark.integration
class TestEmbedErrors:
    """Integration tests for error handling."""

    def test_embed_invalid_base64(self, client: TestClient, invalid_base64: str) -> None:
        """Embed endpoint rejects invalid base64."""
        response = client.post(
            "/embed",
            json={"image": invalid_base64},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "decode_error"

    def test_embed_not_an_image(self, client: TestClient, not_an_image_base64: str) -> None:
        """Embed endpoint rejects non-image data."""
        response = client.post(
            "/embed",
            json={"image": not_an_image_base64},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "invalid_image"

    def test_embed_missing_image_field(self, client: TestClient) -> None:
        """Embed endpoint requires image field."""
        response = client.post(
            "/embed",
            json={},
        )
        assert response.status_code == 422  # Validation error

    def test_embed_unsupported_format_details(
        self, client: TestClient, bmp_image_base64: str
    ) -> None:
        """Unsupported format error includes details."""
        response = client.post(
            "/embed",
            json={"image": bmp_image_base64},
        )
        data = response.json()
        assert "details" in data
        assert "detected_format" in data["details"]
        assert "supported_formats" in data["details"]


# =============================================================================
# Embed Endpoint Integration Tests - Data URL
# =============================================================================


@pytest.mark.integration
class TestEmbedDataUrl:
    """Integration tests for data URL handling."""

    def test_embed_accepts_data_url(self, client: TestClient, sample_image_base64: str) -> None:
        """Embed endpoint accepts data URL format."""
        data_url = f"data:image/jpeg;base64,{sample_image_base64}"
        response = client.post(
            "/embed",
            json={"image": data_url},
        )
        assert response.status_code == 200
