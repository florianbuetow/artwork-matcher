"""
Unit tests for request/response schemas.

Tests Pydantic model validation for all API schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gateway.schemas import (
    BackendInfo,
    BackendsInfo,
    BackendStatus,
    DebugInfo,
    ErrorResponse,
    HealthResponse,
    IdentifyOptions,
    IdentifyRequest,
    IdentifyResponse,
    InfoResponse,
    Match,
    ObjectDetails,
    ObjectListResponse,
    ObjectSummary,
    PipelineInfo,
    TimingInfo,
)


@pytest.mark.unit
class TestIdentifyRequest:
    """Tests for identify request schema."""

    def test_valid_request(self) -> None:
        """Valid request with required fields."""
        request = IdentifyRequest(image="base64encodeddata")

        assert request.image == "base64encodeddata"
        assert request.options is None

    def test_valid_request_with_options(self) -> None:
        """Valid request with all fields."""
        request = IdentifyRequest(
            image="base64encodeddata",
            options=IdentifyOptions(k=10, threshold=0.8),
        )

        assert request.options is not None
        assert request.options.k == 10
        assert request.options.threshold == 0.8

    def test_missing_required_field_raises_error(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IdentifyRequest()  # type: ignore[call-arg]

        assert "image" in str(exc_info.value)

    def test_empty_image_raises_error(self) -> None:
        """Empty image string raises ValidationError (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            IdentifyRequest(image="")

        assert "image" in str(exc_info.value).lower()

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            IdentifyRequest(image="data", unknown_field="value")  # type: ignore[call-arg]

        assert "extra" in str(exc_info.value).lower()


@pytest.mark.unit
class TestIdentifyOptions:
    """Tests for identify options schema."""

    def test_all_fields_optional(self) -> None:
        """All fields are optional with None defaults."""
        options = IdentifyOptions()

        assert options.k is None
        assert options.threshold is None
        assert options.geometric_verification is None
        assert options.include_alternatives is False

    def test_valid_options(self) -> None:
        """Valid options with all fields."""
        options = IdentifyOptions(
            k=5,
            threshold=0.7,
            geometric_verification=True,
            include_alternatives=True,
        )

        assert options.k == 5
        assert options.threshold == 0.7
        assert options.geometric_verification is True
        assert options.include_alternatives is True

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError."""
        with pytest.raises(ValidationError):
            IdentifyOptions(unknown="value")  # type: ignore[call-arg]


@pytest.mark.unit
class TestIdentifyResponse:
    """Tests for identify response schema."""

    def test_valid_response_with_match(self) -> None:
        """Valid response with a match."""
        response = IdentifyResponse(
            success=True,
            match=Match(
                object_id="obj_001",
                confidence=0.89,
                similarity_score=0.92,
                verification_method="geometric",
            ),
            timing=TimingInfo(
                embedding_ms=47.2,
                search_ms=1.3,
                geometric_ms=156.8,
                total_ms=208.4,
            ),
        )

        assert response.success is True
        assert response.match is not None
        assert response.match.object_id == "obj_001"
        assert response.timing.total_ms == 208.4

    def test_valid_response_no_match(self) -> None:
        """Valid response with no match."""
        response = IdentifyResponse(
            success=True,
            match=None,
            message="No matching artwork found",
            timing=TimingInfo(
                embedding_ms=45.1,
                search_ms=1.2,
                geometric_ms=0,
                total_ms=49.8,
            ),
        )

        assert response.success is True
        assert response.match is None
        assert response.message == "No matching artwork found"

    def test_response_with_alternatives(self) -> None:
        """Response with alternative matches."""
        response = IdentifyResponse(
            success=True,
            match=Match(
                object_id="obj_001",
                confidence=0.89,
                similarity_score=0.92,
                verification_method="geometric",
            ),
            alternatives=[
                Match(
                    object_id="obj_002",
                    confidence=0.72,
                    similarity_score=0.85,
                    verification_method="geometric",
                ),
            ],
            timing=TimingInfo(
                embedding_ms=47.2,
                search_ms=1.3,
                geometric_ms=156.8,
                total_ms=208.4,
            ),
        )

        assert response.alternatives is not None
        assert len(response.alternatives) == 1
        assert response.alternatives[0].object_id == "obj_002"


@pytest.mark.unit
class TestMatch:
    """Tests for match schema."""

    def test_valid_match_minimal(self) -> None:
        """Valid match with required fields only."""
        match = Match(
            object_id="obj_001",
            confidence=0.89,
            similarity_score=0.92,
            verification_method="geometric",
        )

        assert match.object_id == "obj_001"
        assert match.name is None
        assert match.artist is None

    def test_valid_match_full(self) -> None:
        """Valid match with all fields."""
        match = Match(
            object_id="obj_001",
            name="Water Lilies",
            artist="Claude Monet",
            year="1906",
            confidence=0.89,
            similarity_score=0.92,
            geometric_score=0.85,
            verification_method="geometric",
            image_url="/objects/obj_001/image",
        )

        assert match.name == "Water Lilies"
        assert match.artist == "Claude Monet"
        assert match.geometric_score == 0.85

    def test_verification_method_literal(self) -> None:
        """Verification method must be valid literal."""
        # Valid methods
        for method in ["geometric", "embedding_only"]:
            match = Match(
                object_id="obj_001",
                confidence=0.89,
                similarity_score=0.92,
                verification_method=method,  # type: ignore[arg-type]
            )
            assert match.verification_method == method

    def test_invalid_verification_method_raises_error(self) -> None:
        """Invalid verification method raises ValidationError."""
        with pytest.raises(ValidationError):
            Match(
                object_id="obj_001",
                confidence=0.89,
                similarity_score=0.92,
                verification_method="invalid_method",  # type: ignore[arg-type]
            )


@pytest.mark.unit
class TestHealthResponse:
    """Tests for health response schema."""

    def test_valid_health_response(self) -> None:
        """Valid health response."""
        response = HealthResponse(
            status="healthy",
            backends=BackendStatus(
                embeddings="healthy",
                search="healthy",
                geometric="healthy",
                storage="healthy",
            ),
        )

        assert response.status == "healthy"
        assert response.backends is not None
        assert response.backends.embeddings == "healthy"

    def test_health_response_without_backends(self) -> None:
        """Health response without backends check."""
        response = HealthResponse(status="healthy")

        assert response.status == "healthy"
        assert response.backends is None

    def test_health_status_literal_validation(self) -> None:
        """Status must be one of the allowed values."""
        for status in ["healthy", "degraded", "unhealthy"]:
            response = HealthResponse(status=status)  # type: ignore[arg-type]
            assert response.status == status

    def test_invalid_status_raises_error(self) -> None:
        """Invalid status value raises ValidationError."""
        with pytest.raises(ValidationError):
            HealthResponse(status="invalid_status")  # type: ignore[arg-type]


@pytest.mark.unit
class TestInfoResponse:
    """Tests for info response schema."""

    def test_valid_info_response(self) -> None:
        """Valid info response with nested models."""
        response = InfoResponse(
            service="gateway",
            version="0.1.0",
            pipeline=PipelineInfo(
                search_k=5,
                similarity_threshold=0.7,
                geometric_verification=True,
                confidence_threshold=0.6,
            ),
            backends=BackendsInfo(
                embeddings=BackendInfo(url="http://localhost:8001", status="healthy"),
                search=BackendInfo(url="http://localhost:8002", status="healthy"),
                geometric=BackendInfo(url="http://localhost:8003", status="healthy"),
            ),
        )

        assert response.service == "gateway"
        assert response.pipeline.search_k == 5
        assert response.backends.embeddings.url == "http://localhost:8001"


@pytest.mark.unit
class TestPipelineInfo:
    """Tests for pipeline info schema."""

    def test_valid_pipeline_info(self) -> None:
        """Valid pipeline info."""
        info = PipelineInfo(
            search_k=5,
            similarity_threshold=0.7,
            geometric_verification=True,
            confidence_threshold=0.6,
        )

        assert info.search_k == 5
        assert info.similarity_threshold == 0.7
        assert info.geometric_verification is True


@pytest.mark.unit
class TestTimingInfo:
    """Tests for timing info schema."""

    def test_valid_timing_info(self) -> None:
        """Valid timing info."""
        timing = TimingInfo(
            embedding_ms=47.2,
            search_ms=1.3,
            geometric_ms=156.8,
            total_ms=208.4,
        )

        assert timing.embedding_ms == 47.2
        assert timing.total_ms == 208.4


@pytest.mark.unit
class TestDebugInfo:
    """Tests for debug info schema."""

    def test_valid_debug_info_minimal(self) -> None:
        """Valid debug info with required fields only."""
        debug = DebugInfo(candidates_considered=4)

        assert debug.candidates_considered == 4
        assert debug.candidates_verified is None
        assert debug.highest_similarity is None

    def test_valid_debug_info_full(self) -> None:
        """Valid debug info with all fields."""
        debug = DebugInfo(
            candidates_considered=4,
            candidates_verified=4,
            embedding_dimension=768,
            highest_similarity=0.43,
            threshold=0.7,
        )

        assert debug.candidates_verified == 4
        assert debug.embedding_dimension == 768


@pytest.mark.unit
class TestObjectListResponse:
    """Tests for object list response schema."""

    def test_valid_object_list(self) -> None:
        """Valid object list response."""
        response = ObjectListResponse(
            objects=[
                ObjectSummary(object_id="obj_001", name="Mona Lisa"),
                ObjectSummary(object_id="obj_002", name="Starry Night"),
            ],
            count=2,
        )

        assert len(response.objects) == 2
        assert response.count == 2
        assert response.objects[0].object_id == "obj_001"

    def test_empty_object_list(self) -> None:
        """Empty object list response."""
        response = ObjectListResponse(objects=[], count=0)

        assert len(response.objects) == 0
        assert response.count == 0


@pytest.mark.unit
class TestObjectDetails:
    """Tests for object details schema."""

    def test_valid_object_details_minimal(self) -> None:
        """Valid object details with required fields only."""
        details = ObjectDetails(object_id="obj_001")

        assert details.object_id == "obj_001"
        assert details.name is None

    def test_valid_object_details_full(self) -> None:
        """Valid object details with all fields."""
        details = ObjectDetails(
            object_id="obj_001",
            name="Water Lilies",
            artist="Claude Monet",
            year="1906",
            description="Part of a series...",
            location="Gallery 3",
            image_url="/objects/obj_001/image",
            indexed_at="2025-01-15T10:30:00Z",
        )

        assert details.name == "Water Lilies"
        assert details.artist == "Claude Monet"
        assert details.location == "Gallery 3"


@pytest.mark.unit
class TestErrorResponse:
    """Tests for error response schema."""

    def test_valid_error_response(self) -> None:
        """Valid error response."""
        response = ErrorResponse(
            error="backend_unavailable",
            message="Embeddings service is not responding",
            details={"backend": "embeddings"},
        )

        assert response.error == "backend_unavailable"
        assert "Embeddings" in response.message
        assert response.details["backend"] == "embeddings"

    def test_error_response_with_empty_details(self) -> None:
        """Error response with empty details dict."""
        response = ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            details={},
        )

        assert response.details == {}
