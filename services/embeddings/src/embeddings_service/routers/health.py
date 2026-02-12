"""
Health check endpoint.

Provides service health status for container orchestration
(Docker health checks, Kubernetes probes).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter

from embeddings_service.core.state import get_app_state
from embeddings_service.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Health semantics:
    - healthy: service can process embedding requests (model and processor loaded)
    - unhealthy: service is running but cannot process embedding requests

    Returns:
        Health status with uptime and system time
    """
    state = get_app_state()

    # System time in yyyy-mm-dd hh:mm format (UTC)
    system_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    status: Literal["healthy", "unhealthy"] = (
        "healthy" if state.model is not None and state.processor is not None else "unhealthy"
    )

    return HealthResponse(
        status=status,
        uptime_seconds=state.uptime_seconds,
        uptime=state.uptime_formatted,
        system_time=system_time,
    )
