"""
Integration tests for storage-backed gateway flows.

These tests validate storage integration for:
- Object image proxy endpoint (/objects/{id}/image)
- Identify pipeline reference fetching from storage
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    import respx
    from fastapi.testclient import TestClient


# Backend service URLs (must match config.yaml)
GEOMETRIC_URL = "http://localhost:8003"


# NOTE:
# Storage health integration tests are intentionally omitted.
# /health response currently includes only embeddings/search/geometric in
# gateway/routers/health.py and gateway/schemas.py (no storage backend key).


# =============================================================================
# Object Image Proxy Integration Tests
# =============================================================================


@pytest.mark.integration
class TestObjectImageProxy:
    """Integration tests for GET /objects/{id}/image."""

    def test_get_object_image_returns_bytes(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Object image proxy returns JPEG bytes from storage."""
        response = client.get("/objects/test_obj/image")

        assert response.status_code == 200
        assert response.content == b"fake-jpeg-bytes"
        assert response.headers["content-type"] == "image/jpeg"

    def test_get_object_image_not_found(
        self,
        client: TestClient,
        mock_storage_missing_references: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Object image proxy returns 404 when storage object is missing."""
        response = client.get("/objects/missing_obj/image")

        assert response.status_code == 404

    def test_get_object_image_storage_unavailable(
        self,
        client: TestClient,
        mock_storage_unavailable: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Object image proxy returns 502 when storage is unavailable."""
        response = client.get("/objects/test_obj/image")

        assert response.status_code == 502


# =============================================================================
# Identify + Storage Integration Tests
# =============================================================================


@pytest.mark.integration
class TestIdentifyWithStorageIntegration:
    """Integration tests for identify pipeline behavior with storage references."""

    def test_identify_pipeline_fetches_references_from_storage(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,
        sample_image_base64: str,
    ) -> None:
        """Identify pipeline fetches storage references and completes geometric batch."""
        geometric_batch_route = mock_identify_pipeline.post(f"{GEOMETRIC_URL}/match/batch").mock(
            return_value=httpx.Response(
                200,
                json={
                    "query_id": "query_001",
                    "query_features": 500,
                    "results": [
                        {
                            "reference_id": "artwork_001",
                            "is_match": True,
                            "confidence": 0.88,
                            "inliers": 30,
                            "inlier_ratio": 0.65,
                        }
                    ],
                    "best_match": {
                        "reference_id": "artwork_001",
                        "confidence": 0.88,
                    },
                    "processing_time_ms": 12.5,
                },
            )
        )

        response = client.post(
            "/identify",
            json={
                "image": sample_image_base64,
                "options": {
                    "geometric_verification": True,
                    "k": 5,
                    "threshold": 0.7,
                },
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data["success"] is True
        assert data["match"] is not None
        assert data["match"]["object_id"] == "artwork_001"

        assert geometric_batch_route.called
        payload = json.loads(geometric_batch_route.calls.last.request.content.decode("utf-8"))
        assert payload["references"][0]["reference_id"] == "artwork_001"
        assert payload["references"][0]["reference_image"] == base64.b64encode(
            b"fake-jpeg-bytes"
        ).decode("ascii")

    def test_identify_degrades_when_storage_unavailable(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,
        sample_image_base64: str,
    ) -> None:
        """Identify pipeline degrades gracefully when storage is unavailable."""
        mock_identify_pipeline["storage_objects"].mock(
            side_effect=httpx.ConnectError("storage unavailable")
        )

        response = client.post(
            "/identify",
            json={
                "image": sample_image_base64,
                "options": {
                    "geometric_verification": True,
                    "k": 5,
                    "threshold": 0.7,
                },
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data["success"] is True
        assert data["geometric_skipped"] is True or data["degraded"] is True

    def test_identify_skips_geometric_when_no_references_found(
        self,
        client: TestClient,
        mock_identify_pipeline: respx.MockRouter,
        sample_image_base64: str,
    ) -> None:
        """Identify pipeline skips geometric when storage has no reference images."""
        mock_identify_pipeline["storage_objects"].mock(
            return_value=httpx.Response(404, json={"error": "not_found"})
        )

        response = client.post(
            "/identify",
            json={
                "image": sample_image_base64,
                "options": {
                    "geometric_verification": True,
                    "k": 5,
                    "threshold": 0.7,
                },
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data["success"] is True
        assert data["geometric_skipped"] is True
        assert data["geometric_skip_reason"] == "no_reference_images"
