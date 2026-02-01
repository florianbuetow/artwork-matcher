"""
Embedding extraction endpoint.

Extracts L2-normalized DINOv2 embeddings from base64-encoded images.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter
from PIL import Image

from embeddings_service.core.exceptions import ServiceError
from embeddings_service.core.state import get_app_state
from embeddings_service.logging import get_logger
from embeddings_service.schemas import EmbedRequest, EmbedResponse

router = APIRouter()

# Supported image formats
SUPPORTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png", "image/webp"})


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
    # Handle potential data URL prefix (e.g., "data:image/jpeg;base64,...")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ServiceError(
            error="decode_error",
            message=f"Invalid Base64 encoding: {e}",
            status_code=400,
            details=None,
        ) from e


def load_and_validate_image(image_bytes: bytes) -> Image.Image:
    """
    Load image from bytes and validate format.

    Args:
        image_bytes: Raw image bytes

    Returns:
        PIL Image in RGB mode

    Raises:
        ServiceError: If image cannot be decoded or format is unsupported
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Verify image integrity
        image.verify()
        # Re-open because verify() consumes the file
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ServiceError(
            error="invalid_image",
            message=f"Failed to decode image: {e}",
            status_code=400,
            details=None,
        ) from e

    # Check format
    image_format = image.format
    if image_format is None:
        raise ServiceError(
            error="invalid_image",
            message="Could not determine image format",
            status_code=400,
            details=None,
        )
    if image_format not in SUPPORTED_FORMATS:
        mime_type = Image.MIME.get(image_format)
        if mime_type is None:
            mime_type = f"unknown/{image_format}"
        raise ServiceError(
            error="unsupported_format",
            message=f"Image format '{mime_type}' is not supported. Use JPEG, PNG, or WebP.",
            status_code=400,
            details={
                "detected_format": mime_type,
                "supported_formats": list(SUPPORTED_MIME_TYPES),
            },
        )

    # Convert to RGB (handles RGBA, grayscale, palette, etc.)
    return image.convert("RGB")


@torch.no_grad()
def extract_dino_embedding(
    image: Image.Image,
    model: Any,
    processor: Any,
    device: torch.device,
) -> np.ndarray:
    """
    Extract L2-normalized embedding from image using DINOv2.

    Args:
        image: PIL Image in RGB mode
        model: DINOv2 model
        processor: HuggingFace image processor
        device: Torch device

    Returns:
        1D numpy array of float32, L2-normalized
    """
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Forward pass
    outputs = model(pixel_values)

    # DINOv2 returns CLS token in last_hidden_state[:, 0]
    # This is the classification token embedding
    cls_embedding = outputs.last_hidden_state[:, 0]  # Shape: [1, embedding_dim]

    # L2 normalize for cosine similarity via inner product
    normalized = F.normalize(cls_embedding, p=2, dim=1)

    # Convert to numpy
    embedding = normalized.squeeze(0).cpu().numpy()

    return embedding.astype(np.float32)


@router.post("/embed", response_model=EmbedResponse)
async def extract_embedding(request: EmbedRequest) -> EmbedResponse:
    """
    Extract a normalized embedding vector from an image.

    The embedding is L2-normalized, suitable for cosine similarity
    via inner product.

    Args:
        request: Request containing base64-encoded image

    Returns:
        Embedding vector with metadata
    """
    logger = get_logger()
    start_time = time.perf_counter()

    # 1. Decode base64
    image_bytes = decode_base64_image(request.image)

    # 2. Load and validate image
    image = load_and_validate_image(image_bytes)

    # 3. Get model from app state
    state = get_app_state()
    if state.model is None or state.processor is None or state.device is None:
        raise ServiceError(
            error="model_error",
            message="Model not loaded",
            status_code=500,
            details=None,
        )

    # 4. Extract embedding
    try:
        embedding = extract_dino_embedding(
            image=image,
            model=state.model,
            processor=state.processor,
            device=state.device,
        )
    except Exception as e:
        logger.exception(
            "Model inference failed",
            extra={"image_id": request.image_id},
        )
        raise ServiceError(
            error="model_error",
            message=f"Model inference failed: {e}",
            status_code=500,
            details=None,
        ) from e

    # 5. Calculate processing time
    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Embedding extracted",
        extra={
            "image_id": request.image_id,
            "dimension": len(embedding),
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return EmbedResponse(
        embedding=embedding.tolist(),
        dimension=len(embedding),
        image_id=request.image_id,
        processing_time_ms=round(processing_time_ms, 2),
    )
