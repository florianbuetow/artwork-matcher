"""Unit tests for ORB feature extractor."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from geometric_service.services.feature_extractor import ORBFeatureExtractor


def create_test_image_with_features(width: int = 200, height: int = 200) -> bytes:
    """Create a test image with detectable features (checkerboard pattern)."""
    # Create checkerboard pattern for feature detection
    img = np.zeros((height, width), dtype=np.uint8)
    block_size = 20
    for i in range(0, height, block_size * 2):
        for j in range(0, width, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.unit
class TestORBFeatureExtractor:
    """Tests for ORBFeatureExtractor."""

    def test_extract_returns_keypoints(self) -> None:
        """Should extract keypoints from image."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        image_bytes = create_test_image_with_features()

        keypoints, descriptors, image_size = extractor.extract(image_bytes)

        assert len(keypoints) > 0
        assert all("x" in kp and "y" in kp for kp in keypoints)

    def test_extract_returns_descriptors(self) -> None:
        """Should return 32-byte descriptors."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        image_bytes = create_test_image_with_features()

        keypoints, descriptors, image_size = extractor.extract(image_bytes)

        assert descriptors is not None
        assert descriptors.shape[1] == 32  # ORB descriptors are 32 bytes

    def test_extract_returns_image_size(self) -> None:
        """Should return correct image size."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        image_bytes = create_test_image_with_features(width=300, height=200)

        keypoints, descriptors, image_size = extractor.extract(image_bytes)

        assert image_size == (300, 200)
