"""
Service information endpoint.

Exposes service configuration and index metadata.
"""

from __future__ import annotations

from fastapi import APIRouter

from search_service.config import get_settings
from search_service.core.state import get_app_state
from search_service.schemas import ConfigInfo, IndexInfo, InfoResponse

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information and configuration.

    Returns:
        Service metadata, index statistics, and configuration
    """
    settings = get_settings()
    state = get_app_state()

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        index=IndexInfo(
            type=settings.faiss.index_type,
            metric=settings.faiss.metric,
            embedding_dimension=settings.faiss.embedding_dimension,
            count=state.index_count,
            is_loaded=state.index_loaded,
        ),
        config=ConfigInfo(
            index_path=settings.index.path,
            metadata_path=settings.index.metadata_path,
            default_k=settings.search.default_k,
        ),
    )
