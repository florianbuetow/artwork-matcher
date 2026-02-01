"""
Unit tests for request/response schemas.

Tests Pydantic model validation for all API schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from embeddings_service.schemas import (
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    ModelInfo,
    PreprocessingInfo,
)


@pytest.mark.unit
class TestEmbedRequest:
    """Tests for embedding request schema."""

    def test_valid_request(self) -> None:
        """Valid request with required fields."""
        request = EmbedRequest(image="base64encodeddata")

        assert request.image == "base64encodeddata"
        assert request.image_id is None  # Optional field

    def test_valid_request_with_image_id(self) -> None:
        """Valid request with all fields."""
        request = EmbedRequest(
            image="base64encodeddata",
            image_id="test_001",
        )

        assert request.image_id == "test_001"

    def test_missing_required_field_raises_error(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EmbedRequest()  # type: ignore[call-arg]

        assert "image" in str(exc_info.value)

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            EmbedRequest(image="data", unknown_field="value")  # type: ignore[call-arg]

        assert "extra" in str(exc_info.value).lower()


@pytest.mark.unit
class TestEmbedResponse:
    """Tests for embedding response schema."""

    def test_valid_response(self) -> None:
        """Valid response with all fields."""
        response = EmbedResponse(
            embedding=[0.1, 0.2, 0.3],
            dimension=3,
            image_id="test_001",
            processing_time_ms=45.2,
        )

        assert response.dimension == 3
        assert len(response.embedding) == 3
        assert response.image_id == "test_001"

    def test_response_with_none_image_id(self) -> None:
        """Response with null image_id."""
        response = EmbedResponse(
            embedding=[0.1, 0.2],
            dimension=2,
            image_id=None,
            processing_time_ms=10.0,
        )

        assert response.image_id is None

    def test_embedding_dimension_not_validated_against_list_length(self) -> None:
        """Schema doesn't validate embedding length matches dimension."""
        # This is intentional - validation happens in business logic
        response = EmbedResponse(
            embedding=[0.1, 0.2],
            dimension=768,  # Doesn't match embedding length
            image_id=None,
            processing_time_ms=45.2,
        )

        assert response.dimension == 768
        assert len(response.embedding) == 2


@pytest.mark.unit
class TestHealthResponse:
    """Tests for health response schema."""

    def test_valid_health_response(self) -> None:
        """Valid health response."""
        response = HealthResponse(
            status="healthy",
            uptime_seconds=123.45,
            uptime="2m 3s",
            system_time="2024-01-01 12:00",
        )

        assert response.status == "healthy"
        assert response.uptime_seconds == 123.45

    def test_health_status_literal_validation(self) -> None:
        """Status must be one of the allowed values."""
        # Valid statuses
        for status in ["healthy", "degraded", "unhealthy"]:
            response = HealthResponse(
                status=status,  # type: ignore[arg-type]
                uptime_seconds=100.0,
                uptime="1m 40s",
                system_time="2024-01-01 12:00",
            )
            assert response.status == status

    def test_invalid_status_raises_error(self) -> None:
        """Invalid status value raises ValidationError."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="invalid_status",  # type: ignore[arg-type]
                uptime_seconds=100.0,
                uptime="1m 40s",
                system_time="2024-01-01 12:00",
            )


@pytest.mark.unit
class TestInfoResponse:
    """Tests for info response schema."""

    def test_valid_info_response(self) -> None:
        """Valid info response with nested models."""
        response = InfoResponse(
            service="embeddings",
            version="0.1.0",
            model=ModelInfo(
                name="facebook/dinov2-base",
                embedding_dimension=768,
                device="cpu",
            ),
            preprocessing=PreprocessingInfo(
                image_size=518,
                normalize=True,
            ),
        )

        assert response.service == "embeddings"
        assert response.model.embedding_dimension == 768
        assert response.preprocessing.image_size == 518


@pytest.mark.unit
class TestModelInfo:
    """Tests for model info schema."""

    def test_valid_model_info(self) -> None:
        """Valid model info."""
        info = ModelInfo(
            name="facebook/dinov2-base",
            embedding_dimension=768,
            device="cuda",
        )

        assert info.name == "facebook/dinov2-base"
        assert info.embedding_dimension == 768
        assert info.device == "cuda"


@pytest.mark.unit
class TestPreprocessingInfo:
    """Tests for preprocessing info schema."""

    def test_valid_preprocessing_info(self) -> None:
        """Valid preprocessing info."""
        info = PreprocessingInfo(
            image_size=518,
            normalize=True,
        )

        assert info.image_size == 518
        assert info.normalize is True


@pytest.mark.unit
class TestErrorResponse:
    """Tests for error response schema."""

    def test_valid_error_response(self) -> None:
        """Valid error response."""
        response = ErrorResponse(
            error="invalid_image",
            message="Failed to decode image",
            details={"format": "unknown"},
        )

        assert response.error == "invalid_image"
        assert "decode" in response.message
        assert response.details["format"] == "unknown"

    def test_error_response_with_empty_details(self) -> None:
        """Error response with empty details dict."""
        response = ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            details={},
        )

        assert response.details == {}
