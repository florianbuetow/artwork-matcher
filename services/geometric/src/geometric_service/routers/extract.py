"""
Feature extraction endpoint.
"""

from __future__ import annotations

import base64
import binascii
import time

from fastapi import APIRouter

from geometric_service.config import get_settings
from geometric_service.core.exceptions import ServiceError
from geometric_service.logging import get_logger
from geometric_service.schemas import (
    ExtractRequest,
    ExtractResponse,
    ImageSize,
    KeypointData,
)
from geometric_service.services.feature_extractor import ORBFeatureExtractor

router = APIRouter()


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to bytes."""
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


@router.post("/extract", response_model=ExtractResponse)
async def extract_features(request: ExtractRequest) -> ExtractResponse:
    """Extract ORB features from an image."""
    logger = get_logger()
    start_time = time.perf_counter()

    settings = get_settings()

    image_bytes = decode_base64_image(request.image)

    max_features = request.max_features or settings.orb.max_features
    extractor = ORBFeatureExtractor(
        max_features=max_features,
        scale_factor=settings.orb.scale_factor,
        n_levels=settings.orb.n_levels,
        edge_threshold=settings.orb.edge_threshold,
        patch_size=settings.orb.patch_size,
        fast_threshold=settings.orb.fast_threshold,
    )

    keypoints, descriptors, image_size = extractor.extract(image_bytes)

    if len(keypoints) < settings.verification.min_features:
        raise ServiceError(
            error="insufficient_features",
            message=f"Only {len(keypoints)} features extracted (minimum: {settings.verification.min_features})",
            status_code=422,
            details={
                "image_id": request.image_id,
                "features_found": len(keypoints),
                "minimum_required": settings.verification.min_features,
            },
        )

    if descriptors is not None:
        descriptors_b64 = base64.b64encode(descriptors.tobytes()).decode("ascii")
    else:
        descriptors_b64 = ""

    kp_list = [
        KeypointData(x=kp["x"], y=kp["y"], size=kp["size"], angle=kp["angle"])
        for kp in keypoints
    ]

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Features extracted",
        extra={
            "image_id": request.image_id,
            "num_features": len(keypoints),
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return ExtractResponse(
        image_id=request.image_id,
        num_features=len(keypoints),
        keypoints=kp_list,
        descriptors=descriptors_b64,
        image_size=ImageSize(width=image_size[0], height=image_size[1]),
        processing_time_ms=round(processing_time_ms, 2),
    )
