"""
Feature matching using brute-force matcher with Lowe's ratio test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BFFeatureMatcher:
    """Match features using brute-force with ratio test."""

    def __init__(self, ratio_threshold: float) -> None:
        """
        Initialize feature matcher.

        Args:
            ratio_threshold: Lowe's ratio test threshold (e.g., 0.75)
        """
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_threshold = ratio_threshold

    def match(
        self,
        desc1: NDArray[np.uint8] | None,
        desc2: NDArray[np.uint8] | None,
    ) -> list[cv2.DMatch]:
        """
        Match descriptors using kNN + Lowe's ratio test.

        Args:
            desc1: Descriptors from first image (N1, 32)
            desc2: Descriptors from second image (N2, 32)

        Returns:
            List of good matches after ratio test filtering.
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Need at least 2 descriptors in desc2 for kNN with k=2
        if len(desc2) < 2:
            return []

        # Find 2 nearest neighbors for ratio test
        try:
            matches = self.bf.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        # Apply Lowe's ratio test
        good_matches: list[cv2.DMatch] = []
        for match_pair in matches:
            # Some matches may have fewer than 2 results
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        return good_matches
