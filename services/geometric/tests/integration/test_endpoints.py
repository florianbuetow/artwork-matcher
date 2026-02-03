"""Integration tests for all endpoints."""

from __future__ import annotations

import pytest

from tests.factories import create_checkerboard_base64, create_non_image_base64


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for /health."""

    def test_health_returns_healthy(self, client) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.integration
class TestInfoEndpoint:
    """Integration tests for /info."""

    def test_info_returns_algorithm_config(self, client) -> None:
        """Info endpoint returns algorithm configuration."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "geometric"
        assert data["algorithm"]["feature_detector"] == "ORB"


@pytest.mark.integration
class TestExtractEndpoint:
    """Integration tests for /extract."""

    def test_extract_checkerboard(self, client) -> None:
        """Extract features from checkerboard image."""
        response = client.post(
            "/extract",
            json={"image": create_checkerboard_base64(200, 200), "image_id": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["num_features"] > 0
        assert len(data["keypoints"]) == data["num_features"]

    def test_extract_invalid_image(self, client) -> None:
        """Extract returns error for invalid image."""
        response = client.post(
            "/extract",
            json={"image": create_non_image_base64()},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_image"


@pytest.mark.integration
class TestMatchEndpoint:
    """Integration tests for /match."""

    def test_match_identical_images(self, client) -> None:
        """Matching identical images returns is_match=True."""
        image = create_checkerboard_base64(200, 200)
        response = client.post(
            "/match",
            json={"query_image": image, "reference_image": image},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_match"] is True
        # Confidence > 0 indicates positive match with geometric verification
        assert data["confidence"] > 0


@pytest.mark.integration
class TestBatchMatchEndpoint:
    """Integration tests for /match/batch."""

    def test_batch_match(self, client) -> None:
        """Batch match returns results for all references."""
        query = create_checkerboard_base64(200, 200, seed=0)
        ref1 = create_checkerboard_base64(200, 200, seed=0)
        ref2 = create_checkerboard_base64(200, 200, seed=42)

        response = client.post(
            "/match/batch",
            json={
                "query_image": query,
                "references": [
                    {"reference_id": "ref_001", "reference_image": ref1},
                    {"reference_id": "ref_002", "reference_image": ref2},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
