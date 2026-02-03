"""
Image generators for performance testing.

Provides functions to generate test images with controlled properties
for measuring ORB feature extraction and geometric matching performance.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def create_noise_image_base64(
    width: int,
    height: int,
    quality: int = 85,
) -> str:
    """
    Generate a JPEG image with random noise as base64.

    Random noise creates images with many detectable features,
    ideal for testing ORB extraction performance.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        quality: JPEG quality (1-100), higher = larger file

    Returns:
        Base64-encoded JPEG image string
    """
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(noise, mode="RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_checkerboard_base64(
    width: int,
    height: int,
    block_size: int = 20,
) -> str:
    """
    Generate a checkerboard image with detectable corner features.

    Checkerboard patterns provide consistent corner features
    that are reliably detected by ORB.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        block_size: Size of checkerboard squares

    Returns:
        Base64-encoded PNG image string
    """
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, block_size * 2):
        for j in range(0, width, block_size * 2):
            img[i : i + block_size, j : j + block_size] = 255
            img[i + block_size : i + block_size * 2, j + block_size : j + block_size * 2] = 255

    pil_image = Image.fromarray(img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_transformed_image_base64(
    base_image_b64: str,
    rotation_deg: float = 0,
    scale: float = 1.0,
) -> str:
    """
    Create a transformed version of an image for matching tests.

    Applies rotation and/or scale to simulate real-world image variations.

    Args:
        base_image_b64: Base64-encoded source image
        rotation_deg: Rotation angle in degrees
        scale: Scale factor (1.0 = no change)

    Returns:
        Base64-encoded transformed PNG image
    """
    image_bytes = base64.b64decode(base_image_b64)
    image = Image.open(BytesIO(image_bytes))

    if rotation_deg != 0:
        image = image.rotate(rotation_deg, expand=True, fillcolor=(128, 128, 128))

    if scale != 1.0:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_image_size_kb(image_base64: str) -> float:
    """
    Calculate the size of a base64-encoded image in kilobytes.

    Args:
        image_base64: Base64-encoded image string

    Returns:
        Size in kilobytes
    """
    image_bytes = base64.b64decode(image_base64)
    return len(image_bytes) / 1024
