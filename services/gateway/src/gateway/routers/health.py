"""
Health check endpoint.

Provides service health status for container orchestration
(Docker health checks, Kubernetes probes).
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from gateway.core.state import get_app_state
from gateway.schemas import BackendStatus, HealthResponse

router = APIRouter()


# nosemgrep: no-default-parameter-values (intentional API query parameter default)
@router.get("/health", response_model=HealthResponse)
async def health_check(
    check_backends: bool = Query(
        default=True,
        description="Whether to check backend service health",
    ),
) -> HealthResponse:
    """
    Check service health.

    Health semantics:
    - healthy: gateway and all critical backends (embeddings/search) are healthy
    - degraded: critical backends are healthy but optional geometric backend is not healthy
    - unhealthy: at least one critical backend is not healthy

    Returns:
        Health status with optional backend status
    """
    state = get_app_state()
    backends: BackendStatus | None = None
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"

    if check_backends:
        # Check all backend services
        embeddings_status = await state.embeddings_client.health_check()
        search_status = await state.search_client.health_check()
        geometric_status = await state.geometric_client.health_check()

        backends = BackendStatus(
            embeddings=embeddings_status,
            search=search_status,
            geometric=geometric_status,
        )

        # Determine overall status
        # Embeddings and search are critical
        if embeddings_status != "healthy" or search_status != "healthy":
            status = "unhealthy"
        # Geometric is optional (graceful degradation)
        elif geometric_status != "healthy":
            status = "degraded"

    return HealthResponse(
        status=status,
        backends=backends,
    )
