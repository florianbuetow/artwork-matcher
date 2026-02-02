"""
Pydantic request/response models for the search service API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# === Request Models ===


class SearchRequest(BaseModel):
    """Request model for POST /search endpoint."""

    model_config = ConfigDict(extra="forbid")

    embedding: list[float] = Field(..., min_length=1)
    """Query embedding vector (must match index dimension)."""

    k: int | None = Field(None, ge=1, le=10000)
    """Maximum results to return. Uses default from config if not specified.

    The actual limit is enforced by config.search.max_k (typically 100).
    The schema limit of 10000 is a safety bound to prevent memory issues.
    """

    threshold: float | None = Field(None, ge=0.0, le=1.0)
    """Minimum similarity score. Uses default from config if not specified."""


class AddRequest(BaseModel):
    """Request model for POST /add endpoint."""

    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(..., min_length=1)
    """Unique identifier for this object."""

    embedding: list[float] = Field(..., min_length=1)
    """Embedding vector (must match index dimension)."""

    metadata: dict[str, Any] | None = None
    """Optional metadata to store with this embedding."""


class SaveRequest(BaseModel):
    """Request model for POST /index/save endpoint."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    """Custom save path (optional, uses config default if not specified)."""


class LoadRequest(BaseModel):
    """Request model for POST /index/load endpoint."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    """Custom load path (optional, uses config default if not specified)."""


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


class IndexInfo(BaseModel):
    """Index information for /info endpoint."""

    type: str
    """FAISS index type (flat, ivf, hnsw)."""

    metric: str
    """Distance metric (inner_product, l2)."""

    embedding_dimension: int
    """Expected vector dimension."""

    count: int
    """Number of vectors in index."""

    is_loaded: bool
    """Whether index is ready for search."""

    load_error: str | None = None
    """Error message if index auto-load failed at startup. None if load succeeded."""


class ConfigInfo(BaseModel):
    """Configuration info for /info endpoint."""

    index_path: str
    """Path to FAISS index file."""

    metadata_path: str
    """Path to metadata JSON file."""

    default_k: int
    """Default number of search results."""


class InfoResponse(BaseModel):
    """Response model for GET /info endpoint."""

    service: str
    """Service name."""

    version: str
    """Service version (semver)."""

    index: IndexInfo
    """Index statistics and configuration."""

    config: ConfigInfo
    """Configuration values."""


class SearchResultItem(BaseModel):
    """A single search result."""

    object_id: str
    """Unique identifier for the matched object."""

    score: float
    """Similarity score (higher = more similar)."""

    rank: int
    """1-indexed rank in results."""

    metadata: dict[str, Any]
    """Associated metadata."""


class SearchResponse(BaseModel):
    """Response model for POST /search endpoint."""

    results: list[SearchResultItem]
    """Ranked list of matches."""

    count: int
    """Number of results returned."""

    query_dimension: int
    """Dimension of query vector (for validation)."""

    processing_time_ms: float
    """Search time in milliseconds."""


class AddResponse(BaseModel):
    """Response model for POST /add endpoint."""

    object_id: str
    """Echo of input object_id."""

    index_position: int
    """Position in the FAISS index."""

    index_count: int
    """Total items in index after add."""


class SaveResponse(BaseModel):
    """Response model for POST /index/save endpoint."""

    index_path: str
    """Path where index was saved."""

    metadata_path: str
    """Path where metadata was saved."""

    count: int
    """Number of items saved."""

    size_bytes: int
    """Size of index file in bytes."""


class LoadResponse(BaseModel):
    """Response model for POST /index/load endpoint."""

    index_path: str
    """Path from which index was loaded."""

    metadata_path: str
    """Path from which metadata was loaded."""

    count: int
    """Number of items loaded."""

    dimension: int
    """Embedding dimension of loaded index."""


class ClearResponse(BaseModel):
    """Response model for DELETE /index endpoint."""

    previous_count: int
    """Number of items before clearing."""

    current_count: int
    """Number of items after clearing (should be 0)."""


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    """Machine-readable error code."""

    message: str
    """Human-readable error description."""

    details: dict[str, Any]
    """Additional error context."""
