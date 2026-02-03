"""
Test data factories for generating test fixtures.

Provides reusable functions for creating test images with features.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def create_checkerboard_image(
    width: int = 200,
    height: int = 200,
    block_size: int = 20,
    seed: int | None = None,
) -> bytes:
    """
    Create a checkerboard image with detectable features.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        block_size: Size of checkerboard squares
        seed: Random seed for reproducibility

    Returns:
        Raw image bytes (PNG format)
    """
    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, block_size * 2):
        for j in range(0, width, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_checkerboard_base64(
    width: int = 200,
    height: int = 200,
    block_size: int = 20,
    seed: int | None = None,
) -> str:
    """Create a checkerboard image as base64."""
    image_bytes = create_checkerboard_image(width, height, block_size, seed)
    return base64.b64encode(image_bytes).decode("ascii")


def create_solid_color_image(
    width: int = 100,
    height: int = 100,
    color: str = "red",
) -> bytes:
    """Create a solid color image (no features)."""
    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_solid_color_base64(
    width: int = 100,
    height: int = 100,
    color: str = "red",
) -> str:
    """Create a solid color image as base64."""
    image_bytes = create_solid_color_image(width, height, color)
    return base64.b64encode(image_bytes).decode("ascii")


def create_invalid_base64() -> str:
    """Create an invalid base64 string for error testing."""
    return "not-valid-base64!!!"


def create_non_image_base64() -> str:
    """Create valid base64 that is not an image."""
    return base64.b64encode(b"not an image file").decode("ascii")
