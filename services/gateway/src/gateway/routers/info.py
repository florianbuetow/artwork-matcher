"""
Service information endpoint.

Exposes service configuration and backend status.
"""

from __future__ import annotations

import contextlib

from fastapi import APIRouter

from gateway.config import get_settings
from gateway.core.state import get_app_state
from gateway.schemas import BackendInfo, BackendsInfo, InfoResponse, PipelineInfo

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
async def get_info() -> InfoResponse:
    """
    Get service information and configuration.

    Returns:
        Service metadata, pipeline configuration, and backend status
    """
    settings = get_settings()
    state = get_app_state()

    # Get backend health and info
    embeddings_status = await state.embeddings_client.health_check()
    search_status = await state.search_client.health_check()
    geometric_status = await state.geometric_client.health_check()

    # Try to get detailed info from backends
    embeddings_info = None
    search_info = None
    geometric_info = None

    with contextlib.suppress(Exception):
        embeddings_info = await state.embeddings_client.get_info()

    with contextlib.suppress(Exception):
        search_info = await state.search_client.get_info()

    with contextlib.suppress(Exception):
        geometric_info = await state.geometric_client.get_info()

    return InfoResponse(
        service=settings.service.name,
        version=settings.service.version,
        pipeline=PipelineInfo(
            search_k=settings.pipeline.search_k,
            similarity_threshold=settings.pipeline.similarity_threshold,
            geometric_verification=settings.pipeline.geometric_verification,
            confidence_threshold=settings.pipeline.confidence_threshold,
        ),
        backends=BackendsInfo(
            embeddings=BackendInfo(
                url=settings.backends.embeddings_url,
                status=embeddings_status,
                info=embeddings_info,
            ),
            search=BackendInfo(
                url=settings.backends.search_url,
                status=search_status,
                info=search_info,
            ),
            geometric=BackendInfo(
                url=settings.backends.geometric_url,
                status=geometric_status,
                info=geometric_info,
            ),
        ),
    )
