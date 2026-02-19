"""
Pydantic request/response models for the storage service API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"]
    """Service health status."""

    uptime_seconds: float
    """Uptime in seconds since service start."""

    uptime: str
    """Human-readable uptime."""

    system_time: str
    """Current system time in yyyy-mm-dd hh:mm format (UTC)."""


class StorageInfo(BaseModel):
    """Storage configuration for /info endpoint."""

    model_config = ConfigDict(extra="forbid")

    path: str
    """Storage directory path."""

    content_type: str
    """Content type for stored objects."""

    object_count: int
    """Number of objects currently stored."""


class InfoResponse(BaseModel):
    """Response model for GET /info endpoint."""

    service: str
    """Service name."""

    version: str
    """Service version."""

    storage: StorageInfo
    """Storage configuration and status."""


class DeleteAllResponse(BaseModel):
    """Response model for DELETE /objects endpoint."""

    model_config = ConfigDict(extra="forbid")

    deleted_count: int
    """Number of objects deleted."""


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    """Machine-readable error code."""

    message: str
    """Human-readable error description."""

    details: dict[str, Any]
    """Additional error context."""
