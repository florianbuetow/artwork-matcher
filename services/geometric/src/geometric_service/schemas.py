"""
Pydantic request/response models for the geometric service API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# === Helper Models ===


class KeypointData(BaseModel):
    """Keypoint data for feature extraction."""

    model_config = ConfigDict(extra="forbid")

    x: float
    """X coordinate of the keypoint."""

    y: float
    """Y coordinate of the keypoint."""

    size: float
    """Size of the keypoint neighborhood."""

    angle: float
    """Orientation of the keypoint in degrees."""


class ImageSize(BaseModel):
    """Image dimensions."""

    model_config = ConfigDict(extra="forbid")

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""


# === Request Models ===


class ExtractRequest(BaseModel):
    """Request model for POST /extract endpoint."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(..., min_length=1)
    """Base64-encoded image data (JPEG, PNG, or WebP)."""

    image_id: str | None = None
    """Optional identifier for logging/tracing."""

    max_features: int | None = None
    """Optional override for maximum features to extract."""


class ReferenceFeatures(BaseModel):
    """Pre-extracted features for reference image."""

    model_config = ConfigDict(extra="forbid")

    keypoints: list[KeypointData]
    """List of keypoint data."""

    descriptors: str
    """Base64-encoded descriptors array."""


class MatchRequest(BaseModel):
    """Request model for POST /match endpoint."""

    model_config = ConfigDict(extra="forbid")

    query_image: str = Field(..., min_length=1)
    """Base64-encoded query image data."""

    reference_image: str | None = None
    """Base64-encoded reference image data."""

    reference_features: ReferenceFeatures | None = None
    """Pre-extracted features for reference image."""

    query_id: str | None = None
    """Optional identifier for query image."""

    reference_id: str | None = None
    """Optional identifier for reference image."""


class ReferenceInput(BaseModel):
    """Reference input for batch matching."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str
    """Identifier for the reference image."""

    reference_image: str | None = None
    """Base64-encoded reference image data."""

    reference_features: ReferenceFeatures | None = None
    """Pre-extracted features for reference image."""


class BatchMatchRequest(BaseModel):
    """Request model for POST /match/batch endpoint."""

    model_config = ConfigDict(extra="forbid")

    query_image: str = Field(..., min_length=1)
    """Base64-encoded query image data."""

    references: list[ReferenceInput]
    """List of reference images/features to match against."""

    query_id: str | None = None
    """Optional identifier for query image."""


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


class AlgorithmInfo(BaseModel):
    """Algorithm configuration for /info endpoint."""

    feature_detector: str
    """Feature detector algorithm name (e.g., ORB)."""

    max_features: int
    """Maximum number of features to extract."""

    matcher: str
    """Matcher algorithm name (e.g., BFMatcher)."""

    matcher_norm: str
    """Norm type for matcher (e.g., HAMMING)."""

    ratio_threshold: float
    """Lowe's ratio test threshold."""

    verification: str
    """Geometric verification method (e.g., RANSAC)."""

    ransac_reproj_threshold: float
    """RANSAC reprojection threshold in pixels."""

    min_inliers: int
    """Minimum inliers required for a valid match."""


class InfoResponse(BaseModel):
    """Response model for GET /info endpoint."""

    service: str
    """Service name."""

    version: str
    """Service version (semver)."""

    algorithm: AlgorithmInfo
    """Algorithm configuration."""


class ExtractResponse(BaseModel):
    """Response model for POST /extract endpoint."""

    image_id: str | None
    """Echo of input image_id."""

    num_features: int
    """Number of features extracted."""

    keypoints: list[KeypointData]
    """List of extracted keypoints."""

    descriptors: str
    """Base64-encoded descriptors array."""

    image_size: ImageSize
    """Original image dimensions."""

    processing_time_ms: float
    """Server-side processing time in milliseconds."""


class MatchResponse(BaseModel):
    """Response model for POST /match endpoint."""

    is_match: bool
    """Whether the images match."""

    confidence: float
    """Match confidence score (0.0 to 1.0)."""

    inliers: int
    """Number of inlier matches after RANSAC."""

    total_matches: int
    """Total number of feature matches before RANSAC."""

    inlier_ratio: float
    """Ratio of inliers to total matches."""

    query_features: int
    """Number of features in query image."""

    reference_features: int
    """Number of features in reference image."""

    homography: list[list[float]] | None = None
    """3x3 homography matrix if match found."""

    query_id: str | None = None
    """Echo of input query_id."""

    reference_id: str | None = None
    """Echo of input reference_id."""

    processing_time_ms: float
    """Server-side processing time in milliseconds."""


class BatchMatchResult(BaseModel):
    """Individual result in batch match response."""

    reference_id: str
    """Identifier for the reference image."""

    is_match: bool
    """Whether the images match."""

    confidence: float
    """Match confidence score (0.0 to 1.0)."""

    inliers: int
    """Number of inlier matches after RANSAC."""

    inlier_ratio: float
    """Ratio of inliers to total matches."""


class BestMatch(BaseModel):
    """Best match information in batch match response."""

    reference_id: str
    """Identifier of the best matching reference."""

    confidence: float
    """Confidence score of the best match."""


class BatchMatchResponse(BaseModel):
    """Response model for POST /match/batch endpoint."""

    query_id: str | None
    """Echo of input query_id."""

    query_features: int
    """Number of features in query image."""

    results: list[BatchMatchResult]
    """List of match results for each reference."""

    best_match: BestMatch | None
    """Information about the best match, if any."""

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
