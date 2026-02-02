"""
Index management endpoints.

Provides endpoints for adding embeddings, saving/loading the index,
and clearing the index.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from search_service.config import get_settings
from search_service.core.exceptions import ServiceError
from search_service.core.state import get_app_state
from search_service.logging import get_logger
from search_service.schemas import (
    AddRequest,
    AddResponse,
    ClearResponse,
    LoadRequest,
    LoadResponse,
    SaveRequest,
    SaveResponse,
)
from search_service.services.faiss_index import (
    DimensionMismatchError,
    IndexLoadError,
    IndexSaveError,
    InvalidEmbeddingError,
)

router = APIRouter()


def validate_index_path(user_path: str, allowed_base: Path) -> Path:
    """
    Validate that a user-provided path is within the allowed directory.

    Prevents path traversal attacks by ensuring the resolved path
    is a child of the allowed base directory.

    Args:
        user_path: The path provided by the user
        allowed_base: The base directory that paths must be within

    Returns:
        The validated, resolved path

    Raises:
        ServiceError: If the path is outside the allowed directory
    """
    # Resolve both paths to eliminate .. and symlinks
    resolved_path = Path(user_path).resolve()
    allowed_resolved = allowed_base.resolve()

    # Check if the resolved path is within the allowed directory
    try:
        resolved_path.relative_to(allowed_resolved)
    except ValueError as e:
        raise ServiceError(
            error="path_not_allowed",
            message="Path must be within the allowed index directory",
            status_code=400,
            details={
                "allowed_base": str(allowed_resolved),
                "requested_path": str(resolved_path),
            },
        ) from e

    return resolved_path


@router.post("/add", response_model=AddResponse, status_code=201)
async def add_embedding(request: AddRequest) -> AddResponse:
    """
    Add an embedding to the index.

    The embedding is added to memory only. Call POST /index/save to persist.

    Args:
        request: Add request with object_id, embedding, and optional metadata

    Returns:
        Confirmation with index position and count

    Raises:
        ServiceError: If index is not loaded or embedding is invalid
    """
    settings = get_settings()
    state = get_app_state()
    logger = get_logger()

    # Check if index is loaded
    if not state.index_loaded or state.faiss_index is None:
        raise ServiceError(
            error="index_not_loaded",
            message="Index is not loaded. Call POST /index/load or wait for startup.",
            status_code=503,
            details={"index_path": settings.index.path},
        )

    try:
        position = state.faiss_index.add(
            object_id=request.object_id,
            embedding=request.embedding,
            metadata=request.metadata,
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

    logger.info(
        "Added embedding to index",
        extra={
            "object_id": request.object_id,
            "index_position": position,
            "index_count": state.faiss_index.count,
        },
    )

    return AddResponse(
        object_id=request.object_id,
        index_position=position,
        index_count=state.faiss_index.count,
    )


@router.post("/index/save", response_model=SaveResponse)
async def save_index(request: SaveRequest) -> SaveResponse:
    """
    Persist the current index to disk.

    Args:
        request: Optional custom save path

    Returns:
        Confirmation with paths, count, and size

    Raises:
        ServiceError: If index is not loaded or save fails
    """
    settings = get_settings()
    state = get_app_state()
    logger = get_logger()

    # Check if index is loaded
    if not state.index_loaded or state.faiss_index is None:
        raise ServiceError(
            error="index_not_loaded",
            message="Index is not loaded. Nothing to save.",
            status_code=503,
            details={"index_path": settings.index.path},
        )

    # Determine paths
    # The allowed base directory is configurable, defaults to parent of index path
    if settings.index.allowed_path_base is not None:
        allowed_base = Path(settings.index.allowed_path_base)
    else:
        allowed_base = Path(settings.index.path).parent

    if request.path is not None:
        # Validate custom path is within allowed directory
        index_path = validate_index_path(request.path, allowed_base)
        metadata_path = index_path.with_suffix(".json")
    else:
        index_path = Path(settings.index.path)
        metadata_path = Path(settings.index.metadata_path)

    try:
        size_bytes = state.faiss_index.save(index_path, metadata_path)
    except IndexSaveError as e:
        raise ServiceError(
            error="save_failed",
            message=str(e),
            status_code=500,
            details={"index_path": str(index_path)},
        ) from e

    logger.info(
        "Saved index to disk",
        extra={
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "count": state.faiss_index.count,
            "size_bytes": size_bytes,
        },
    )

    return SaveResponse(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        count=state.faiss_index.count,
        size_bytes=size_bytes,
    )


@router.post("/index/load", response_model=LoadResponse)
async def load_index(request: LoadRequest) -> LoadResponse:
    """
    Load an index from disk.

    Args:
        request: Optional custom load path

    Returns:
        Confirmation with paths, count, and dimension

    Raises:
        ServiceError: If index is not initialized or load fails
    """
    settings = get_settings()
    state = get_app_state()
    logger = get_logger()

    # Check if index wrapper exists
    if state.faiss_index is None:
        raise ServiceError(
            error="index_not_loaded",
            message="Index wrapper not initialized. This should not happen.",
            status_code=503,
            details={},
        )

    # Determine paths
    # The allowed base directory is configurable, defaults to parent of index path
    if settings.index.allowed_path_base is not None:
        allowed_base = Path(settings.index.allowed_path_base)
    else:
        allowed_base = Path(settings.index.path).parent

    if request.path is not None:
        # Validate custom path is within allowed directory
        index_path = validate_index_path(request.path, allowed_base)
        metadata_path = index_path.with_suffix(".json")
    else:
        index_path = Path(settings.index.path)
        metadata_path = Path(settings.index.metadata_path)

    try:
        state.faiss_index.load(index_path, metadata_path)
    except IndexLoadError as e:
        raise ServiceError(
            error="load_failed",
            message=str(e),
            status_code=500,
            details={"index_path": str(index_path)},
        ) from e
    except DimensionMismatchError as e:
        raise ServiceError(
            error="dimension_mismatch",
            message=str(e),
            status_code=400,
            details={"expected": e.expected, "received": e.received},
        ) from e

    logger.info(
        "Loaded index from disk",
        extra={
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "count": state.faiss_index.count,
            "dimension": state.faiss_index.dimension,
        },
    )

    return LoadResponse(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        count=state.faiss_index.count,
        dimension=state.faiss_index.dimension,
    )


@router.delete("/index", response_model=ClearResponse)
async def clear_index() -> ClearResponse:
    """
    Clear the index (remove all vectors and metadata).

    Returns:
        Previous and current counts

    Raises:
        ServiceError: If index is not loaded
    """
    settings = get_settings()
    state = get_app_state()
    logger = get_logger()

    # Check if index is loaded
    if not state.index_loaded or state.faiss_index is None:
        raise ServiceError(
            error="index_not_loaded",
            message="Index is not loaded. Nothing to clear.",
            status_code=503,
            details={"index_path": settings.index.path},
        )

    previous_count = state.faiss_index.clear()

    logger.info(
        "Cleared index",
        extra={
            "previous_count": previous_count,
            "current_count": state.faiss_index.count,
        },
    )

    return ClearResponse(
        previous_count=previous_count,
        current_count=state.faiss_index.count,
    )
