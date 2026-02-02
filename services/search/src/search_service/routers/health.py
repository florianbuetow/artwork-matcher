"""
Health check endpoint.

Provides service health status for container orchestration
(Docker health checks, Kubernetes probes).
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from search_service.core.state import get_app_state
from search_service.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns:
        Health status with uptime and system time
    """
    state = get_app_state()

    # System time in yyyy-mm-dd hh:mm format
    system_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    return HealthResponse(
        status="healthy",
        uptime_seconds=state.uptime_seconds,
        uptime=state.uptime_formatted,
        system_time=system_time,
    )
