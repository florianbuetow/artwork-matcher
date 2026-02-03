"""
Health check endpoint.

Provides service health status for container orchestration.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from geometric_service.core.state import get_app_state
from geometric_service.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health."""
    state = get_app_state()
    system_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    return HealthResponse(
        status="healthy",
        uptime_seconds=state.uptime_seconds,
        uptime=state.uptime_formatted,
        system_time=system_time,
    )
