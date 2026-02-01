"""
Pydantic request/response models for the embeddings service API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# === Request Models ===


class EmbedRequest(BaseModel):
    """Request model for POST /embed endpoint."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(..., min_length=1)
    """Base64-encoded image data (JPEG, PNG, or WebP)."""

    image_id: str | None = None
    """Optional identifier for logging/tracing."""


# === Response Models ===


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"]
    """Service health status."""

    uptime_seconds: float
    """Uptime in seconds since service start."""

    uptime: str
    """Human-readable uptime (e.g., "2d 3h 15m 42s")."""

    system_time: str
    """Current system time in yyyy-mm-dd hh:mm format (UTC)."""


class ModelInfo(BaseModel):
    """Model configuration info for /info endpoint."""

    name: str
    """HuggingFace model identifier."""

    embedding_dimension: int
    """Output embedding vector dimension."""

    device: str
    """Compute device (cpu, cuda, mps)."""


class PreprocessingInfo(BaseModel):
    """Preprocessing configuration info for /info endpoint."""

    image_size: int
    """Input image resize target."""

    normalize: bool
    """Whether ImageNet normalization is applied."""


class InfoResponse(BaseModel):
    """Response model for GET /info endpoint."""

    service: str
    """Service name."""

    version: str
    """Service version (semver)."""

    model: ModelInfo
    """Model configuration."""

    preprocessing: PreprocessingInfo
    """Preprocessing configuration."""


class EmbedResponse(BaseModel):
    """Response model for POST /embed endpoint."""

    embedding: list[float]
    """L2-normalized embedding vector."""

    dimension: int
    """Embedding dimension (for client validation)."""

    image_id: str | None
    """Echo of input image_id."""

    processing_time_ms: float
    """Server-side processing time in milliseconds."""


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    """Machine-readable error code."""

    message: str
    """Human-readable error description."""

    details: dict[str, Any]
    """Additional error context."""
