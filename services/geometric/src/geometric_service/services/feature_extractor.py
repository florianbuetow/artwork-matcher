"""
ORB feature extraction from images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from geometric_service.core.exceptions import ServiceError


class ORBFeatureExtractor:
    """Extract ORB features from images."""

    def __init__(
        self,
        max_features: int,
        scale_factor: float,
        n_levels: int,
        edge_threshold: int,
        patch_size: int,
        fast_threshold: int,
    ) -> None:
        """
        Initialize ORB feature extractor.

        Args:
            max_features: Maximum number of features to retain
            scale_factor: Pyramid decimation ratio (> 1.0)
            n_levels: Number of pyramid levels
            edge_threshold: Border pixels excluded from detection
            patch_size: Size of patch used for descriptor
            fast_threshold: FAST corner detection threshold
        """
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )

    def extract(
        self, image_bytes: bytes
    ) -> tuple[list[dict[str, float]], NDArray[np.uint8] | None, tuple[int, int]]:
        """
        Extract ORB features from image bytes.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, or WebP)

        Returns:
            Tuple of:
            - keypoints: List of {x, y, size, angle}
            - descriptors: numpy array (N, 32) or None if no features
            - image_size: (width, height)

        Raises:
            ServiceError: If image cannot be decoded
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ServiceError(
                error="invalid_image",
                message="Failed to decode image data",
                status_code=400,
                details=None,
            )

        # Convert to grayscale for ORB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get image dimensions (height, width, channels)
        height, width = gray.shape[:2]
        image_size = (width, height)

        # Detect keypoints and compute descriptors
        cv_keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # Convert keypoints to serializable format
        keypoints: list[dict[str, float]] = []
        for kp in cv_keypoints:
            keypoints.append(
                {
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                }
            )

        return keypoints, descriptors, image_size

    def keypoints_to_cv(self, keypoints: list[dict[str, float]]) -> list[cv2.KeyPoint]:
        """
        Convert serialized keypoints back to OpenCV KeyPoint objects.

        Args:
            keypoints: List of {x, y, size, angle}

        Returns:
            List of cv2.KeyPoint objects
        """
        cv_keypoints = []
        for kp in keypoints:
            cv_keypoints.append(
                cv2.KeyPoint(
                    kp["x"],
                    kp["y"],
                    kp["size"],
                    kp["angle"],
                )
            )
        return cv_keypoints
