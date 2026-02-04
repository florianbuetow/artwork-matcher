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


def create_transformed_image_base64(
    base_image_b64: str,
    rotation_deg: float = 0,
    scale: float = 1.0,
    crop_ratio: float = 1.0,
) -> str:
    """
    Create a transformed version of an image.

    Applies rotation, scale, and/or cropping to simulate real-world
    variations like photos taken at angles or different distances.

    Args:
        base_image_b64: Base64-encoded source image
        rotation_deg: Rotation angle in degrees (counterclockwise)
        scale: Scale factor (1.0 = no change, >1 = enlarge, <1 = shrink)
        crop_ratio: Fraction of image to keep (1.0 = no crop, 0.5 = center 50%)

    Returns:
        Base64-encoded transformed PNG image
    """
    image_bytes = base64.b64decode(base_image_b64)
    image = Image.open(BytesIO(image_bytes))

    # Convert to RGB if grayscale for consistent handling
    if image.mode == "L":
        image = image.convert("RGB")

    # Apply rotation
    if rotation_deg != 0:
        image = image.rotate(rotation_deg, expand=True, fillcolor=(128, 128, 128))

    # Apply scale
    if scale != 1.0:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Apply center crop
    if crop_ratio < 1.0:
        crop_width = int(image.width * crop_ratio)
        crop_height = int(image.height * crop_ratio)
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_noise_image(
    width: int = 200,
    height: int = 200,
    seed: int | None = None,
) -> bytes:
    """
    Create a random noise image with many detectable features.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility

    Returns:
        Raw image bytes (PNG format)
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    pil_image = Image.fromarray(noise, mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_noise_image_base64(
    width: int = 200,
    height: int = 200,
    seed: int | None = None,
) -> str:
    """Create a random noise image as base64."""
    image_bytes = create_noise_image(width, height, seed)
    return base64.b64encode(image_bytes).decode("ascii")


def create_artwork_simulation_image(
    width: int = 300,
    height: int = 300,
    seed: int = 0,
) -> bytes:
    """
    Create a simulated artwork image with distinct visual features.

    Generates an image with geometric shapes, gradients, and patterns
    that simulates a painting or artwork with identifiable features.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility

    Returns:
        Raw image bytes (PNG format)
    """
    np.random.seed(seed)

    # Start with a colored background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_color = np.random.randint(50, 200, 3)
    img[:, :] = bg_color

    # Add some geometric shapes
    for _ in range(np.random.randint(5, 15)):
        shape_type = np.random.choice(["rect", "circle", "line"])
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))

        if shape_type == "rect":
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
            img[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = color
        elif shape_type == "circle":
            # Simple circle approximation using numpy
            cy, cx = np.random.randint(30, height - 30), np.random.randint(30, width - 30)
            radius = np.random.randint(10, 40)
            y, x = np.ogrid[:height, :width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            img[mask] = color
        else:  # line
            # Draw a thick diagonal line
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            # Simple line by drawing points
            num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
            if num_points > 1:
                xs = np.linspace(x1, x2, num_points).astype(int)
                ys = np.linspace(y1, y2, num_points).astype(int)
                for px, py in zip(xs, ys, strict=True):
                    if 0 <= py < height and 0 <= px < width:
                        # Thick line
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                ny, nx = py + dy, px + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    img[ny, nx] = color

    pil_image = Image.fromarray(img, mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_artwork_simulation_base64(
    width: int = 300,
    height: int = 300,
    seed: int = 0,
) -> str:
    """Create a simulated artwork image as base64."""
    image_bytes = create_artwork_simulation_image(width, height, seed)
    return base64.b64encode(image_bytes).decode("ascii")
