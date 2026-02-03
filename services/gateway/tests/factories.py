"""
Test data factories for generating test fixtures.

Provides reusable functions for creating test images, mock responses,
and other test data with consistent, predictable outputs.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image


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


def create_mock_search_result(
    object_id: str = "object_001",
    score: float = 0.92,
    rank: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a mock search result dictionary.

    Args:
        object_id: Unique identifier for the matched object
        score: Similarity score (0-1)
        rank: Result ranking position
        metadata: Additional metadata about the match

    Returns:
        Dictionary representing a search result
    """
    if metadata is None:
        metadata = {"name": "Test Artwork", "artist": "Test Artist"}

    return {
        "object_id": object_id,
        "score": score,
        "rank": rank,
        "metadata": metadata,
    }


def create_mock_embedding(
    dimension: int = 768,
    value: float = 0.1,
) -> list[float]:
    """
    Create a mock embedding vector.

    Args:
        dimension: Embedding dimension (768 for DINOv2-base)
        value: Fill value for all dimensions

    Returns:
        List of floats representing the embedding
    """
    return [value] * dimension


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
