"""Image processing utilities."""

from __future__ import annotations

import base64
import binascii

import numpy as np

from geometric_service.core.exceptions import ServiceError


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decode base64 string to bytes.

    Args:
        base64_string: Base64-encoded image data

    Returns:
        Raw image bytes

    Raises:
        ServiceError: If base64 decoding fails
    """
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        return base64.b64decode(base64_string)
    except binascii.Error as e:
        raise ServiceError(
            error="decode_error",
            message=f"Invalid Base64 encoding: {e}",
            status_code=400,
            details=None,
        ) from e


def decode_descriptors(descriptors_b64: str, num_features: int) -> np.ndarray:
    """
    Decode base64 descriptors to numpy array.

    Args:
        descriptors_b64: Base64-encoded descriptor data
        num_features: Number of features (rows)

    Returns:
        Numpy array of shape (num_features, 32)

    Raises:
        ServiceError: If decoding fails
    """
    try:
        desc_bytes = base64.b64decode(descriptors_b64)
        return np.frombuffer(desc_bytes, dtype=np.uint8).reshape(num_features, 32)
    except Exception as e:
        raise ServiceError(
            error="invalid_features",
            message=f"Failed to decode descriptors: {e}",
            status_code=400,
            details=None,
        ) from e
