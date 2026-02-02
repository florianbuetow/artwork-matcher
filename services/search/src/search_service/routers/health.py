"""
Health check endpoint.

Provides service health status for container orchestration
(Docker health checks, Kubernetes probes).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter

from search_service.core.state import get_app_state
from search_service.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns health status based on:
    - "healthy": Service running, index loaded or empty by design
    - "degraded": Service running but index auto-load failed (data corruption, etc.)
    - "unhealthy": Service not functioning

    Returns:
        Health status with uptime and system time
    """
    state = get_app_state()

    # System time in yyyy-mm-dd hh:mm format
    system_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    # Determine health status
    # Degraded if index files existed but failed to load (corruption, etc.)
    status: Literal["healthy", "degraded"] = (
        "degraded" if state.index_load_error is not None else "healthy"
    )

    return HealthResponse(
        status=status,
        uptime_seconds=state.uptime_seconds,
        uptime=state.uptime_formatted,
        system_time=system_time,
    )
