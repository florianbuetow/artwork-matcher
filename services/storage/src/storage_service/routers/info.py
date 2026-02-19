"""
Service information endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter

from storage_service.config import get_settings
from storage_service.core.exceptions import ServiceError
from storage_service.core.state import get_app_state
from storage_service.schemas import InfoResponse, StorageInfo

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """Get service information and configuration."""
    settings = get_settings()
    state = get_app_state()
    blob_store = state.blob_store
    if blob_store is None:
        raise ServiceError(
            error="service_unavailable",
            message="Blob store is not initialized",
            status_code=503,
            details={},
        )

    try:
        object_count = blob_store.count()
    except OSError as exc:
        raise ServiceError(
            error="storage_count_failed",
            message="Failed to count stored objects",
            status_code=503,
            details={
                "exception_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        ) from exc

    storage_info = StorageInfo(
        path=settings.storage.path,
        content_type=settings.storage.content_type,
        object_count=object_count,
    )

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        storage=storage_info,
    )
