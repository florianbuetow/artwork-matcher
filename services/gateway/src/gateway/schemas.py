"""
Pydantic request/response models for the gateway service API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# === Request Models ===


class IdentifyOptions(BaseModel):
    """Options for the identification request."""

    model_config = ConfigDict(extra="forbid")

    k: int | None = None
    """Number of candidates to consider (overrides config)."""

    threshold: float | None = None
    """Minimum similarity threshold (overrides config)."""

    geometric_verification: bool | None = None
    """Enable geometric verification (overrides config)."""

    include_alternatives: bool = False
    """Include runner-up matches in response."""


class IdentifyRequest(BaseModel):
    """Request model for POST /identify endpoint."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(..., min_length=1)
    """Base64-encoded visitor photo."""

    options: IdentifyOptions | None = None
    """Optional request options."""


# === Response Models ===


class BackendStatus(BaseModel):
    """Backend service status for health check."""

    embeddings: str
    search: str
    geometric: str


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"]
    """Service health status."""

    backends: BackendStatus | None = None
    """Backend service status (if check_backends=true)."""


class BackendInfo(BaseModel):
    """Backend service info for /info endpoint."""

    url: str
    status: str
    info: dict[str, Any] | None = None


class BackendsInfo(BaseModel):
    """All backend services info."""

    embeddings: BackendInfo
    search: BackendInfo
    geometric: BackendInfo


class PipelineInfo(BaseModel):
    """Pipeline configuration for /info endpoint."""

    search_k: int
    similarity_threshold: float
    geometric_verification: bool
    confidence_threshold: float


class InfoResponse(BaseModel):
    """Response model for GET /info endpoint."""

    service: str
    """Service name."""

    version: str
    """Service version."""

    pipeline: PipelineInfo
    """Pipeline configuration."""

    backends: BackendsInfo
    """Backend service information."""


class Match(BaseModel):
    """A matched artwork."""

    object_id: str
    """Unique object identifier."""

    name: str | None = None
    """Artwork name."""

    artist: str | None = None
    """Artist name."""

    year: str | None = None
    """Year or date range."""

    confidence: float
    """Overall confidence score (0-1)."""

    similarity_score: float
    """Embedding similarity score (0-1)."""

    geometric_score: float | None = None
    """Geometric verification score (if performed)."""

    verification_method: Literal["geometric", "embedding_only"]
    """Method used for verification."""

    image_url: str | None = None
    """URL to retrieve the reference image."""


class TimingInfo(BaseModel):
    """Processing time breakdown."""

    embedding_ms: float
    """Time for embedding extraction."""

    search_ms: float
    """Time for vector search."""

    geometric_ms: float
    """Time for geometric verification (0 if skipped)."""

    total_ms: float
    """Total processing time."""


class DebugInfo(BaseModel):
    """Debug information for /identify response."""

    candidates_considered: int
    """Number of candidates from search."""

    candidates_verified: int | None = None
    """Number geometrically verified."""

    embedding_dimension: int | None = None
    """Embedding vector dimension."""

    highest_similarity: float | None = None
    """Highest similarity score (if no match)."""

    threshold: float | None = None
    """Similarity threshold used."""


class IdentifyResponse(BaseModel):
    """Response model for POST /identify endpoint."""

    success: bool
    """Whether the request was processed successfully."""

    match: Match | None = None
    """Best matching artwork, or None if no match found."""

    alternatives: list[Match] | None = None
    """Other strong matches (if include_alternatives=true)."""

    message: str | None = None
    """Status message (e.g., 'No matching artwork found')."""

    timing: TimingInfo
    """Processing time breakdown."""

    debug: DebugInfo | None = None
    """Debug information."""


class ObjectSummary(BaseModel):
    """Summary info for object listing."""

    object_id: str
    name: str | None = None
    artist: str | None = None
    year: str | None = None


class ObjectListResponse(BaseModel):
    """Response model for GET /objects endpoint."""

    objects: list[ObjectSummary]
    """List of objects."""

    count: int
    """Total count."""


class ObjectDetails(BaseModel):
    """Detailed object information."""

    object_id: str
    name: str | None = None
    artist: str | None = None
    year: str | None = None
    description: str | None = None
    location: str | None = None
    image_url: str | None = None
    indexed_at: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    """Machine-readable error code."""

    message: str
    """Human-readable error description."""

    details: dict[str, Any]
    """Additional error context."""
