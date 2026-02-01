"""
Test data factories for generating test fixtures.

Provides reusable functions for creating test images, embeddings,
and other test data with consistent, predictable outputs.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_test_image(
    width: int = 100,
    height: int = 100,
    color: str = "red",
    format: str = "JPEG",
) -> bytes:
    """
    Create a test image as bytes.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: Fill color (any PIL color name)
        format: Image format (JPEG, PNG, WEBP)

    Returns:
        Raw image bytes
    """
    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def create_test_image_base64(
    width: int = 100,
    height: int = 100,
    color: str = "red",
    format: str = "JPEG",
) -> str:
    """
    Create a base64-encoded test image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: Fill color (any PIL color name)
        format: Image format (JPEG, PNG, WEBP)

    Returns:
        Base64-encoded image string
    """
    image_bytes = create_test_image(width, height, color, format)
    return base64.b64encode(image_bytes).decode("ascii")


def create_random_embedding(
    dimension: int = 768,
    normalized: bool = True,
    seed: int | None = None,
) -> list[float]:
    """
    Create a random embedding vector.

    Args:
        dimension: Embedding dimension (768 for DINOv2-base)
        normalized: Whether to L2-normalize the embedding
        seed: Random seed for reproducibility

    Returns:
        List of floats representing the embedding
    """
    if seed is not None:
        np.random.seed(seed)

    embedding: NDArray[np.float32] = np.random.randn(dimension).astype(np.float32)
    if normalized:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    return embedding.tolist()


def create_invalid_base64() -> str:
    """Create an invalid base64 string for error testing."""
    return "not-valid-base64!!!"


def create_non_image_base64() -> str:
    """Create valid base64 that is not an image."""
    return base64.b64encode(b"not an image file").decode("ascii")


def create_bmp_image_base64() -> str:
    """Create a BMP image (unsupported format) as base64."""
    image = Image.new("RGB", (10, 10), color="yellow")
    buffer = BytesIO()
    image.save(buffer, format="BMP")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
