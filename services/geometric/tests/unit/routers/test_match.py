"""Unit tests for match endpoints."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def create_checkerboard_image(seed: int = 0) -> str:
    """Create a checkerboard image with features as base64."""
    np.random.seed(seed)
    img = np.zeros((200, 200), dtype=np.uint8)
    block_size = 20
    for i in range(0, 200, block_size * 2):
        for j in range(0, 200, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255
    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_mock_settings() -> MagicMock:
    """Create mock settings for tests."""
    settings = MagicMock()
    settings.service.name = "geometric"
    settings.service.version = "0.1.0"
    settings.orb.max_features = 1000
    settings.orb.scale_factor = 1.2
    settings.orb.n_levels = 8
    settings.orb.edge_threshold = 31
    settings.orb.patch_size = 31
    settings.orb.fast_threshold = 20
    settings.matching.ratio_threshold = 0.75
    settings.ransac.reproj_threshold = 5.0
    settings.ransac.max_iters = 2000
    settings.ransac.confidence = 0.995
    settings.verification.min_features = 10
    settings.verification.min_matches = 4
    settings.verification.min_inliers = 4
    return settings


@pytest.mark.unit
class TestMatchEndpoint:
    """Tests for POST /match."""

    def test_match_same_image_returns_match(self) -> None:
        """Matching same image should return is_match=True."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            image = create_checkerboard_image()
            response = client.post(
                "/match",
                json={"query_image": image, "reference_image": image},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["is_match"] is True
            assert data["inliers"] > 0

    def test_match_returns_processing_time(self) -> None:
        """Match endpoint should return processing time."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            image = create_checkerboard_image()
            response = client.post(
                "/match",
                json={"query_image": image, "reference_image": image},
            )

            data = response.json()
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] > 0


@pytest.mark.unit
class TestBatchMatchEndpoint:
    """Tests for POST /match/batch."""

    def test_batch_match_returns_results(self) -> None:
        """Batch match should return results for all references."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            query = create_checkerboard_image(seed=0)
            ref1 = create_checkerboard_image(seed=0)
            ref2 = create_checkerboard_image(seed=42)

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
            assert data["results"][0]["reference_id"] == "ref_001"
