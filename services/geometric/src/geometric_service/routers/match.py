"""
Geometric matching endpoints.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from geometric_service.config import get_settings
from geometric_service.core.exceptions import ServiceError
from geometric_service.logging import get_logger
from geometric_service.schemas import (
    BatchMatchRequest,
    BatchMatchResponse,
    BatchMatchResult,
    BestMatch,
    MatchRequest,
    MatchResponse,
)
from geometric_service.services.feature_extractor import ORBFeatureExtractor
from geometric_service.services.feature_matcher import BFFeatureMatcher
from geometric_service.services.geometric_verifier import RANSACVerifier
from geometric_service.utils.image import decode_base64_image, decode_descriptors

router = APIRouter()


@router.post("/match", response_model=MatchResponse)
async def match_images(request: MatchRequest) -> MatchResponse:
    """Geometrically verify two images."""
    logger = get_logger()
    start_time = time.perf_counter()

    settings = get_settings()

    extractor = ORBFeatureExtractor(
        max_features=settings.orb.max_features,
        scale_factor=settings.orb.scale_factor,
        n_levels=settings.orb.n_levels,
        edge_threshold=settings.orb.edge_threshold,
        patch_size=settings.orb.patch_size,
        fast_threshold=settings.orb.fast_threshold,
    )

    query_bytes = decode_base64_image(request.query_image)
    query_kp, query_desc, _ = extractor.extract(query_bytes)
    query_cv_kp = extractor.keypoints_to_cv(query_kp)

    if request.reference_image is not None:
        ref_bytes = decode_base64_image(request.reference_image)
        ref_kp, ref_desc, _ = extractor.extract(ref_bytes)
        ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
    elif request.reference_features is not None:
        ref_kp = [kp.model_dump() for kp in request.reference_features.keypoints]
        ref_desc = decode_descriptors(
            request.reference_features.descriptors,
            len(request.reference_features.keypoints),
        )
        ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
    else:
        raise ServiceError(
            error="validation_error",
            message="Either reference_image or reference_features must be provided",
            status_code=400,
            details=None,
        )

    if len(query_kp) < settings.verification.min_features:
        raise ServiceError(
            error="insufficient_features",
            message=(
                f"Query has only {len(query_kp)} features "
                f"(minimum: {settings.verification.min_features})"
            ),
            status_code=422,
            details={"query_features": len(query_kp)},
        )

    if len(ref_kp) < settings.verification.min_features:
        raise ServiceError(
            error="insufficient_features",
            message=(
                f"Reference has only {len(ref_kp)} features "
                f"(minimum: {settings.verification.min_features})"
            ),
            status_code=422,
            details={"reference_features": len(ref_kp)},
        )

    matcher = BFFeatureMatcher(ratio_threshold=settings.matching.ratio_threshold)
    matches = matcher.match(query_desc, ref_desc)

    verifier = RANSACVerifier(
        reproj_threshold=settings.ransac.reproj_threshold,
        max_iters=settings.ransac.max_iters,
        confidence=settings.ransac.confidence,
        min_inliers=settings.verification.min_inliers,
    )
    result = verifier.verify(query_cv_kp, ref_cv_kp, matches)

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Match completed",
        extra={
            "query_id": request.query_id,
            "reference_id": request.reference_id,
            "is_match": result["is_match"],
            "inliers": result["inliers"],
            "confidence": result["confidence"],
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return MatchResponse(
        is_match=result["is_match"],
        confidence=result["confidence"],
        inliers=result["inliers"],
        total_matches=result["total_matches"],
        inlier_ratio=result["inlier_ratio"],
        query_features=len(query_kp),
        reference_features=len(ref_kp),
        homography=result["homography"],
        query_id=request.query_id,
        reference_id=request.reference_id,
        processing_time_ms=round(processing_time_ms, 2),
    )


@router.post("/match/batch", response_model=BatchMatchResponse)
async def batch_match(request: BatchMatchRequest) -> BatchMatchResponse:
    """Verify query against multiple references."""
    logger = get_logger()
    start_time = time.perf_counter()

    settings = get_settings()

    extractor = ORBFeatureExtractor(
        max_features=settings.orb.max_features,
        scale_factor=settings.orb.scale_factor,
        n_levels=settings.orb.n_levels,
        edge_threshold=settings.orb.edge_threshold,
        patch_size=settings.orb.patch_size,
        fast_threshold=settings.orb.fast_threshold,
    )

    query_bytes = decode_base64_image(request.query_image)
    query_kp, query_desc, _ = extractor.extract(query_bytes)
    query_cv_kp = extractor.keypoints_to_cv(query_kp)

    matcher = BFFeatureMatcher(ratio_threshold=settings.matching.ratio_threshold)
    verifier = RANSACVerifier(
        reproj_threshold=settings.ransac.reproj_threshold,
        max_iters=settings.ransac.max_iters,
        confidence=settings.ransac.confidence,
        min_inliers=settings.verification.min_inliers,
    )

    results: list[BatchMatchResult] = []
    best_match: BestMatch | None = None
    best_confidence = 0.0

    for ref in request.references:
        try:
            if ref.reference_image is not None:
                ref_bytes = decode_base64_image(ref.reference_image)
                ref_kp, ref_desc, _ = extractor.extract(ref_bytes)
                ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
            elif ref.reference_features is not None:
                ref_kp = [kp.model_dump() for kp in ref.reference_features.keypoints]
                ref_desc = decode_descriptors(
                    ref.reference_features.descriptors,
                    len(ref.reference_features.keypoints),
                )
                ref_cv_kp = extractor.keypoints_to_cv(ref_kp)
            else:
                results.append(
                    BatchMatchResult(
                        reference_id=ref.reference_id,
                        is_match=False,
                        confidence=0.0,
                        inliers=0,
                        inlier_ratio=0.0,
                    )
                )
                continue

            matches = matcher.match(query_desc, ref_desc)
            result = verifier.verify(query_cv_kp, ref_cv_kp, matches)

            results.append(
                BatchMatchResult(
                    reference_id=ref.reference_id,
                    is_match=result["is_match"],
                    confidence=result["confidence"],
                    inliers=result["inliers"],
                    inlier_ratio=result["inlier_ratio"],
                )
            )

            if result["is_match"] and result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_match = BestMatch(
                    reference_id=ref.reference_id,
                    confidence=result["confidence"],
                )

        except ServiceError as e:
            logger.warning(
                "Failed to process reference in batch match",
                extra={
                    "reference_id": ref.reference_id,
                    "error": e.error,
                    "message": e.message,
                },
            )
            results.append(
                BatchMatchResult(
                    reference_id=ref.reference_id,
                    is_match=False,
                    confidence=0.0,
                    inliers=0,
                    inlier_ratio=0.0,
                )
            )

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Batch match completed",
        extra={
            "query_id": request.query_id,
            "num_references": len(request.references),
            "best_match": best_match.reference_id if best_match else None,
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )

    return BatchMatchResponse(
        query_id=request.query_id,
        query_features=len(query_kp),
        results=results,
        best_match=best_match,
        processing_time_ms=round(processing_time_ms, 2),
    )
