"""
RANSAC-based geometric verification.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def calculate_confidence(inliers: int, inlier_ratio: float) -> float:
    """
    Combine inlier count and ratio into confidence score.

    - High inliers + high ratio = high confidence
    - High inliers + low ratio = medium confidence
    - Low inliers + high ratio = low confidence

    Args:
        inliers: Number of RANSAC inliers
        inlier_ratio: Ratio of inliers to total matches

    Returns:
        Confidence score between 0 and 1
    """
    # Normalize inlier count (saturates at 100)
    inlier_score = min(inliers / 100.0, 1.0)

    # Weight: 70% inlier count, 30% inlier ratio
    confidence = 0.7 * inlier_score + 0.3 * inlier_ratio

    return round(confidence, 2)


class RANSACVerifier:
    """Verify geometric consistency using RANSAC homography."""

    def __init__(
        self,
        reproj_threshold: float,
        max_iters: int,
        confidence: float,
        min_inliers: int,
    ) -> None:
        """
        Initialize RANSAC verifier.

        Args:
            reproj_threshold: Maximum reprojection error (pixels) to count as inlier
            max_iters: Maximum RANSAC iterations
            confidence: Required confidence in result
            min_inliers: Minimum inliers to declare match
        """
        self.reproj_threshold = reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.min_inliers = min_inliers

    def verify(
        self,
        kp1: list[cv2.KeyPoint],
        kp2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ) -> dict[str, Any]:
        """
        Verify geometric consistency between matched keypoints.

        Args:
            kp1: Keypoints from query image
            kp2: Keypoints from reference image
            matches: Feature matches between images

        Returns:
            Dictionary with:
            - is_match: bool
            - inliers: int
            - total_matches: int
            - inlier_ratio: float
            - homography: list[list[float]] | None
            - confidence: float
        """
        total_matches = len(matches)

        # Need at least 4 points for homography estimation
        if total_matches < 4:
            return {
                "is_match": False,
                "inliers": 0,
                "total_matches": total_matches,
                "inlier_ratio": 0.0,
                "homography": None,
                "confidence": 0.0,
            }

        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography with RANSAC
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=self.reproj_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        # Count inliers
        if mask is None:
            inliers = 0
        else:
            inliers = int(mask.ravel().sum())

        # Calculate inlier ratio
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0.0

        # Determine if match
        is_match = inliers >= self.min_inliers

        # Convert homography to list for JSON serialization
        homography: list[list[float]] | None = None
        if H is not None and is_match:
            homography = H.tolist()

        # Calculate confidence score
        conf = calculate_confidence(inliers, inlier_ratio)

        return {
            "is_match": is_match,
            "inliers": inliers,
            "total_matches": total_matches,
            "inlier_ratio": round(inlier_ratio, 4),
            "homography": homography,
            "confidence": conf,
        }
