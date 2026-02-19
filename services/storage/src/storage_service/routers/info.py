"""
Service information endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter

from storage_service.config import get_settings
from storage_service.core.state import get_app_state
from storage_service.schemas import InfoResponse, StorageInfo

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """Get service information and configuration."""
    settings = get_settings()
    state = get_app_state()

    storage_info = StorageInfo(
        path=settings.storage.path,
        content_type=settings.storage.content_type,
        object_count=state.blob_store.count() if state.blob_store else 0,
    )

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        storage=storage_info,
    )
