"""Unit tests for extract endpoint."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def create_checkerboard_image() -> str:
    """Create a checkerboard image with features as base64."""
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
    settings.verification.min_features = 10
    settings.verification.min_inliers = 10
    return settings


@pytest.mark.unit
class TestExtractEndpoint:
    """Tests for POST /extract."""

    def test_extract_returns_200(self) -> None:
        """Extract endpoint returns 200 OK for valid image."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/extract",
                json={"image": create_checkerboard_image(), "image_id": "test_001"},
            )

            assert response.status_code == 200

    def test_extract_returns_keypoints(self) -> None:
        """Extract endpoint returns keypoints."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/extract",
                json={"image": create_checkerboard_image()},
            )
            data = response.json()

            assert "keypoints" in data
            assert "descriptors" in data
            assert "num_features" in data
            assert data["num_features"] > 0

    def test_extract_invalid_image(self) -> None:
        """Extract endpoint returns 400 for invalid image."""
        with (
            patch("geometric_service.config.get_settings") as mock_settings,
            patch("geometric_service.app.lifespan"),
            patch("geometric_service.routers.health.get_app_state"),
        ):
            mock_settings.return_value = create_mock_settings()

            from geometric_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/extract",
                json={"image": base64.b64encode(b"not an image").decode()},
            )

            assert response.status_code == 400
            assert response.json()["error"] == "invalid_image"
