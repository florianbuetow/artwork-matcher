"""Unit tests for RANSAC geometric verifier."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from geometric_service.services.geometric_verifier import RANSACVerifier, calculate_confidence


@pytest.fixture
def ransac_verifier() -> RANSACVerifier:
    """Provide a configured RANSAC verifier for testing."""
    return RANSACVerifier(
        reproj_threshold=5.0,
        max_iters=2000,
        confidence=0.995,
        min_inliers=10,
    )


@pytest.mark.unit
class TestCalculateConfidence:
    """Tests for confidence score calculation."""

    def test_high_inliers_high_ratio(self) -> None:
        """High inliers + high ratio = high confidence."""
        confidence = calculate_confidence(inliers=100, inlier_ratio=0.8)
        assert confidence >= 0.9

    def test_low_inliers_high_ratio(self) -> None:
        """Low inliers + high ratio = low confidence."""
        confidence = calculate_confidence(inliers=10, inlier_ratio=0.9)
        assert confidence < 0.5

    def test_confidence_bounded(self) -> None:
        """Confidence should be between 0 and 1."""
        for inliers in [0, 10, 50, 100, 200]:
            for ratio in [0.0, 0.3, 0.5, 0.8, 1.0]:
                conf = calculate_confidence(inliers, ratio)
                assert 0.0 <= conf <= 1.0


@pytest.mark.unit
class TestRANSACVerifier:
    """Tests for RANSACVerifier."""

    def test_verify_with_no_matches(self, ransac_verifier: RANSACVerifier) -> None:
        """Should return no match for empty matches."""
        result = ransac_verifier.verify(
            kp1=[],
            kp2=[],
            matches=[],
        )

        assert result["is_match"] is False
        assert result["inliers"] == 0

    def test_verify_with_few_matches(self, ransac_verifier: RANSACVerifier) -> None:
        """Should return no match for too few matches."""
        # Create 3 keypoints (below minimum 4 needed for homography)
        kp1 = [cv2.KeyPoint(i * 10.0, i * 10.0, 10.0) for i in range(3)]
        kp2 = [cv2.KeyPoint(i * 10.0, i * 10.0, 10.0) for i in range(3)]
        matches = [cv2.DMatch(i, i, 0) for i in range(3)]

        result = ransac_verifier.verify(kp1, kp2, matches)

        assert result["is_match"] is False

    def test_verify_returns_expected_fields(self, ransac_verifier: RANSACVerifier) -> None:
        """Should return all expected fields in result."""
        result = ransac_verifier.verify(kp1=[], kp2=[], matches=[])

        assert "is_match" in result
        assert "inliers" in result
        assert "total_matches" in result
        assert "inlier_ratio" in result
        assert "homography" in result
        assert "confidence" in result
