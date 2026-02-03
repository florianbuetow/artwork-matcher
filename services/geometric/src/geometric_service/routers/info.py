"""
Service information endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter

from geometric_service.config import get_settings
from geometric_service.schemas import AlgorithmInfo, InfoResponse

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """Get service information and configuration."""
    settings = get_settings()

    algorithm = AlgorithmInfo(
        feature_detector="ORB",
        max_features=settings.orb.max_features,
        matcher="BFMatcher",
        matcher_norm="HAMMING",
        ratio_threshold=settings.matching.ratio_threshold,
        verification="RANSAC",
        ransac_reproj_threshold=settings.ransac.reproj_threshold,
        min_inliers=settings.verification.min_inliers,
    )

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        algorithm=algorithm,
    )
