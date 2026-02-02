"""
Search endpoint for vector similarity search.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from search_service.config import get_settings
from search_service.core.exceptions import ServiceError
from search_service.core.state import get_app_state
from search_service.schemas import SearchRequest, SearchResponse, SearchResultItem
from search_service.services.faiss_index import (
    DimensionMismatchError,
    InvalidEmbeddingError,
)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search for similar vectors in the index.

    Args:
        request: Search request with embedding, k, and threshold

    Returns:
        Ranked list of matching objects with scores

    Raises:
        ServiceError: If index is not loaded, empty, or embedding is invalid
    """
    settings = get_settings()
    state = get_app_state()

    # Check if index is loaded
    if not state.index_loaded or state.faiss_index is None:
        raise ServiceError(
            error="index_not_loaded",
            message="Index is not loaded. Call POST /index/load or wait for startup.",
            status_code=503,
            details={"index_path": settings.index.path},
        )

    # Check if index is empty
    if state.faiss_index.is_empty:
        raise ServiceError(
            error="index_empty",
            message="Index is empty. Add embeddings before searching.",
            status_code=422,
            details={"count": 0},
        )

    # Use config defaults if not specified
    k = request.k if request.k is not None else settings.search.default_k
    threshold = (
        request.threshold if request.threshold is not None else settings.search.default_threshold
    )

    # Perform search
    start_time = time.perf_counter()

    try:
        results = state.faiss_index.search(
            embedding=request.embedding,
            k=k,
            threshold=threshold,
        )
    except DimensionMismatchError as e:
        raise ServiceError(
            error="dimension_mismatch",
            message=str(e),
            status_code=400,
            details={"expected": e.expected, "received": e.received},
        ) from e
    except InvalidEmbeddingError as e:
        raise ServiceError(
            error="invalid_embedding",
            message=str(e),
            status_code=400,
            details={"reason": e.reason},
        ) from e

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Convert to response format
    result_items = [
        SearchResultItem(
            object_id=r.object_id,
            score=r.score,
            rank=r.rank,
            metadata=r.metadata,
        )
        for r in results
    ]

    return SearchResponse(
        results=result_items,
        count=len(result_items),
        query_dimension=len(request.embedding),
        processing_time_ms=round(elapsed_ms, 3),
    )
