"""
Service information endpoint.

Exposes service configuration and metadata.
"""

from __future__ import annotations

from fastapi import APIRouter

from embeddings_service.config import get_settings
from embeddings_service.core.state import get_app_state
from embeddings_service.schemas import InfoResponse, ModelInfo, PreprocessingInfo

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information and configuration.

    Returns:
        Service metadata and configuration
    """
    settings = get_settings()
    state = get_app_state()

    # Get the resolved device from app state
    device_str = str(state.device) if state.device else settings.model.device

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        model=ModelInfo(
            name=settings.model.name,
            embedding_dimension=settings.model.embedding_dimension,
            device=device_str,
        ),
        preprocessing=PreprocessingInfo(
            image_size=settings.preprocessing.image_size,
            normalize=settings.preprocessing.normalize,
        ),
    )
