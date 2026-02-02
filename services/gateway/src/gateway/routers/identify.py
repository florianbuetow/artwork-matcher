"""
Artwork identification endpoint.

Orchestrates the identification pipeline:
1. Extract embedding from visitor photo
2. Search for similar artworks
3. Optionally verify with geometric matching
4. Return best match with confidence
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx
from fastapi import APIRouter

from gateway.config import get_settings
from gateway.core.exceptions import BackendError
from gateway.core.state import get_app_state
from gateway.logging import get_logger
from gateway.schemas import (
    DebugInfo,
    IdentifyOptions,
    IdentifyRequest,
    IdentifyResponse,
    Match,
    TimingInfo,
)

if TYPE_CHECKING:
    from gateway.clients.search import SearchResult

router = APIRouter()


def calculate_confidence(
    similarity: float,
    geometric_score: float | None,
    geometric_enabled: bool,
) -> float:
    """
    Calculate overall match confidence.

    - If geometric verification passed: weight both scores
    - If geometric verification failed: reduce confidence
    - If geometric not performed: use similarity with penalty
    """
    if geometric_score is not None:
        if geometric_score > 0.5:
            # Geometric confirmed: high confidence
            return 0.6 * similarity + 0.4 * geometric_score
        else:
            # Geometric rejected: low confidence despite similarity
            return 0.3 * similarity + 0.2 * geometric_score
    elif geometric_enabled:
        # Geometric was supposed to run but didn't
        return similarity * 0.7  # Penalty for missing verification
    else:
        # Geometric intentionally skipped
        return similarity * 0.85  # Small penalty


def build_match(
    candidate: SearchResult,
    geometric_score: float | None,
    geometric_enabled: bool,
) -> Match:
    """Build a Match object from search result and geometric verification."""
    confidence = calculate_confidence(
        candidate.score,
        geometric_score,
        geometric_enabled,
    )

    return Match(
        object_id=candidate.object_id,
        name=candidate.metadata.get("name"),
        artist=candidate.metadata.get("artist"),
        year=candidate.metadata.get("year"),
        confidence=round(confidence, 3),
        similarity_score=round(candidate.score, 3),
        geometric_score=round(geometric_score, 3) if geometric_score is not None else None,
        verification_method="geometric" if geometric_score is not None else "embedding_only",
        image_url=f"/objects/{candidate.object_id}/image",
    )


@router.post("/identify", response_model=IdentifyResponse)
async def identify_artwork(request: IdentifyRequest) -> IdentifyResponse:
    """
    Identify artwork in a visitor photo.

    Pipeline:
    1. Extract embedding from visitor photo
    2. Search for similar artworks in the index
    3. Optionally verify with geometric matching
    4. Return best match with confidence score
    """
    logger = get_logger()
    settings = get_settings()
    state = get_app_state()

    timing: dict[str, float] = {}
    start = time.perf_counter()

    # Resolve options (request overrides config defaults)
    options = request.options if request.options is not None else IdentifyOptions()
    k = options.k if options.k is not None else settings.pipeline.search_k
    threshold = (
        options.threshold
        if options.threshold is not None
        else settings.pipeline.similarity_threshold
    )
    do_geometric = (
        options.geometric_verification
        if options.geometric_verification is not None
        else settings.pipeline.geometric_verification
    )
    include_alternatives = options.include_alternatives if options.include_alternatives else False

    logger.info(
        "Starting identification pipeline",
        extra={
            "k": k,
            "threshold": threshold,
            "geometric_verification": do_geometric,
        },
    )

    # Step 1: Extract embedding (embed() raises BackendError on empty/invalid)
    t0 = time.perf_counter()
    embedding = await state.embeddings_client.embed(request.image)
    timing["embedding_ms"] = (time.perf_counter() - t0) * 1000

    logger.debug(
        "Embedding extracted",
        extra={"dimension": len(embedding), "time_ms": timing["embedding_ms"]},
    )

    # Step 2: Search for candidates
    t0 = time.perf_counter()
    candidates = await state.search_client.search(
        embedding=embedding,
        k=k,
        threshold=threshold,
    )
    timing["search_ms"] = (time.perf_counter() - t0) * 1000

    logger.debug(
        "Search completed",
        extra={"candidates": len(candidates), "time_ms": timing["search_ms"]},
    )

    # No candidates found
    if not candidates:
        timing["geometric_ms"] = 0.0
        timing["total_ms"] = (time.perf_counter() - start) * 1000

        return IdentifyResponse(
            success=True,
            match=None,
            message="No matching artwork found with sufficient similarity",
            timing=TimingInfo(**timing),
            debug=DebugInfo(
                candidates_considered=0,
                threshold=threshold,
            ),
        )

    # Step 3: Geometric verification (optional)
    geometric_scores: dict[str, float] = {}
    timing["geometric_ms"] = 0.0

    if do_geometric:
        t0 = time.perf_counter()
        try:
            # Build references list for batch matching
            # Note: In a full implementation, we'd load reference images from storage
            # For now, we'll skip if geometric service is unavailable
            geometric_status = await state.geometric_client.health_check()

            if geometric_status == "healthy":
                # We need reference images - for now log a warning
                # In production, you'd load images from data.objects_path
                logger.warning(
                    "Geometric verification requires reference images - "
                    "skipping (not implemented in this version)"
                )
            else:
                logger.warning("Geometric service unavailable, using embedding only")

        except (BackendError, httpx.HTTPError) as e:
            logger.warning(
                "Geometric verification unavailable",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
        except Exception as e:
            # Unexpected error - log at ERROR level but allow graceful degradation
            logger.error(
                "Unexpected error during geometric verification",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )

        timing["geometric_ms"] = (time.perf_counter() - t0) * 1000

    # Step 4: Calculate confidence and select best match
    matches: list[Match] = []
    for candidate in candidates:
        geo_score = geometric_scores.get(candidate.object_id)
        match = build_match(candidate, geo_score, do_geometric)

        # Filter by confidence threshold
        if match.confidence >= settings.pipeline.confidence_threshold:
            matches.append(match)

    # Sort by confidence descending
    matches.sort(key=lambda m: m.confidence, reverse=True)

    timing["total_ms"] = (time.perf_counter() - start) * 1000

    # Build response
    best_match = matches[0] if matches else None
    alternatives = matches[1:] if include_alternatives and len(matches) > 1 else None

    if not best_match:
        return IdentifyResponse(
            success=True,
            match=None,
            message="No matching artwork found with sufficient confidence",
            timing=TimingInfo(**timing),
            debug=DebugInfo(
                candidates_considered=len(candidates),
                candidates_verified=len(geometric_scores) if do_geometric else None,
                highest_similarity=candidates[0].score if candidates else None,
                threshold=threshold,
            ),
        )

    logger.info(
        "Identification complete",
        extra={
            "match": best_match.object_id,
            "confidence": best_match.confidence,
            "total_ms": timing["total_ms"],
        },
    )

    return IdentifyResponse(
        success=True,
        match=best_match,
        alternatives=alternatives,
        timing=TimingInfo(**timing),
        debug=DebugInfo(
            candidates_considered=len(candidates),
            candidates_verified=len(geometric_scores) if do_geometric else None,
            embedding_dimension=len(embedding),
        ),
    )
