"""Unit tests for ORB feature extractor."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from geometric_service.core.exceptions import ServiceError
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

    def test_extract_raises_service_error_on_invalid_image(self) -> None:
        """Should raise ServiceError for invalid image data."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )
        invalid_bytes = b"not an image"

        with pytest.raises(ServiceError) as exc_info:
            extractor.extract(invalid_bytes)

        assert exc_info.value.error == "invalid_image"

    def test_keypoints_to_cv_converts_correctly(self) -> None:
        """Should convert serialized keypoints to cv2.KeyPoint objects."""
        extractor = ORBFeatureExtractor(
            max_features=500,
            scale_factor=1.2,
            n_levels=8,
            edge_threshold=31,
            patch_size=31,
            fast_threshold=20,
        )

        serialized_kps = [
            {"x": 10.5, "y": 20.3, "size": 5.0, "angle": 45.0},
            {"x": 30.1, "y": 40.2, "size": 7.0, "angle": 90.0},
        ]

        cv_kps = extractor.keypoints_to_cv(serialized_kps)

        assert len(cv_kps) == 2
        assert cv_kps[0].pt[0] == pytest.approx(10.5)
        assert cv_kps[0].pt[1] == pytest.approx(20.3)
        assert cv_kps[0].size == pytest.approx(5.0)
        assert cv_kps[0].angle == pytest.approx(45.0)
