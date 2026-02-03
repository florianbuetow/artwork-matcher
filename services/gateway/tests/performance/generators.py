"""
Image generators for performance testing.

Provides functions to generate test images with controlled properties:
- Solid color images for basic tests
- Noise images for testing with different input characteristics
- Target size images that approximate a specific file size
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def create_solid_image_base64(
    width: int,
    height: int,
    color: tuple[int, int, int] = (128, 128, 128),
    format: str = "JPEG",
) -> str:
    """
    Generate a solid color JPEG image as base64.

    Solid color images compress very well, resulting in small file sizes.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple
        format: Image format (JPEG, PNG)

    Returns:
        Base64-encoded image string
    """
    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format=format, quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_noise_image_base64(
    width: int,
    height: int,
    quality: int = 85,
) -> str:
    """
    Generate a JPEG image with random noise as base64.

    Random noise creates images that don't compress well,
    resulting in larger file sizes compared to solid colors.
    More realistic for testing image processing pipelines.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        quality: JPEG quality (1-100), higher = larger file

    Returns:
        Base64-encoded JPEG image string
    """
    # Generate random RGB noise
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(noise, mode="RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_gradient_image_base64(
    width: int,
    height: int,
    start_color: tuple[int, int, int] = (0, 0, 0),
    end_color: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """
    Generate a gradient image as base64.

    Gradient images have moderate compressibility and represent
    a middle ground between solid colors and noise.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        start_color: Starting RGB color
        end_color: Ending RGB color

    Returns:
        Base64-encoded JPEG image string
    """
    # Create gradient array
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):
        gradient[:, :, i] = np.linspace(start_color[i], end_color[i], width, dtype=np.uint8)

    image = Image.fromarray(gradient, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_target_size_image_base64(
    target_kb: int,
    max_iterations: int = 20,
) -> str:
    """
    Generate a JPEG image that approximates a target file size.

    Uses binary search on JPEG quality and image dimensions to
    find parameters that produce an image close to the target size.

    Args:
        target_kb: Target file size in kilobytes
        max_iterations: Maximum iterations for size tuning

    Returns:
        Base64-encoded JPEG image string
    """
    target_bytes = target_kb * 1024

    # Estimate initial dimensions based on target size
    # Rough heuristic: noise images compress to ~0.3-0.5 bytes per pixel at q=85
    estimated_pixels = target_bytes * 3  # ~0.33 bytes per pixel
    dimension = int(np.sqrt(estimated_pixels))
    dimension = max(100, min(dimension, 4000))  # Clamp to reasonable range

    # Binary search on quality to hit target size
    quality_low = 1
    quality_high = 100
    best_image_b64 = ""
    best_size_diff = float("inf")

    for _ in range(max_iterations):
        quality = (quality_low + quality_high) // 2

        # Generate noise image
        noise = np.random.randint(0, 256, (dimension, dimension, 3), dtype=np.uint8)
        image = Image.fromarray(noise, mode="RGB")

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        image_bytes = buffer.getvalue()
        current_size = len(image_bytes)

        # Track best result
        size_diff = abs(current_size - target_bytes)
        if size_diff < best_size_diff:
            best_size_diff = size_diff
            best_image_b64 = base64.b64encode(image_bytes).decode("ascii")

        # Adjust search based on size comparison
        if current_size < target_bytes:
            # Need larger file: increase quality or dimensions
            if quality >= 95:
                # Quality maxed out, increase dimensions
                dimension = int(dimension * 1.2)
                dimension = min(dimension, 4000)
                quality_low = 1
                quality_high = 100
            else:
                quality_low = quality + 1
        else:
            # Need smaller file: decrease quality
            quality_high = quality - 1

        # Check if we're close enough (within 10%)
        if size_diff < target_bytes * 0.1:
            break

    return best_image_b64


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


def create_test_image_variations(
    dimensions: list[int],
    include_formats: bool = True,
) -> dict[str, str]:
    """
    Create a set of test images with various properties.

    Args:
        dimensions: List of image dimensions to generate
        include_formats: Whether to include different image types (solid, noise, gradient)

    Returns:
        Dictionary mapping descriptive names to base64 images
    """
    images: dict[str, str] = {}

    for dim in dimensions:
        # Solid color images
        images[f"solid_{dim}x{dim}"] = create_solid_image_base64(dim, dim)

        if include_formats:
            # Noise images (more realistic)
            images[f"noise_{dim}x{dim}"] = create_noise_image_base64(dim, dim)

            # Gradient images
            images[f"gradient_{dim}x{dim}"] = create_gradient_image_base64(dim, dim)

    return images
