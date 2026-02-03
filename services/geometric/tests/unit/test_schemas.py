"""
Unit tests for request/response schemas.

Tests Pydantic model validation for all API schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from geometric_service.schemas import (
    AlgorithmInfo,
    BatchMatchRequest,
    BatchMatchResponse,
    BatchMatchResult,
    BestMatch,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    ImageSize,
    InfoResponse,
    KeypointData,
    MatchRequest,
    MatchResponse,
    ReferenceFeatures,
    ReferenceInput,
)


@pytest.mark.unit
class TestKeypointData:
    """Tests for keypoint data schema."""

    def test_valid_keypoint(self) -> None:
        """Valid keypoint with all fields."""
        keypoint = KeypointData(
            x=100.5,
            y=200.3,
            size=10.0,
            angle=45.0,
        )

        assert keypoint.x == 100.5
        assert keypoint.y == 200.3
        assert keypoint.size == 10.0
        assert keypoint.angle == 45.0

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            KeypointData(
                x=100.0,
                y=200.0,
                size=10.0,
                angle=45.0,
                response=1.0,  # type: ignore[call-arg]
            )

        assert "extra" in str(exc_info.value).lower()


@pytest.mark.unit
class TestImageSize:
    """Tests for image size schema."""

    def test_valid_image_size(self) -> None:
        """Valid image size."""
        size = ImageSize(width=640, height=480)

        assert size.width == 640
        assert size.height == 480


@pytest.mark.unit
class TestExtractRequest:
    """Tests for extract request schema."""

    def test_valid_request(self) -> None:
        """Valid request with required fields."""
        request = ExtractRequest(image="base64encodeddata")

        assert request.image == "base64encodeddata"
        assert request.image_id is None
        assert request.max_features is None

    def test_valid_request_with_image_id(self) -> None:
        """Valid request with optional image_id."""
        request = ExtractRequest(
            image="base64encodeddata",
            image_id="test_001",
        )

        assert request.image_id == "test_001"

    def test_valid_request_with_max_features(self) -> None:
        """Valid request with optional max_features override."""
        request = ExtractRequest(
            image="base64encodeddata",
            max_features=500,
        )

        assert request.max_features == 500

    def test_missing_required_field_raises_error(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest()  # type: ignore[call-arg]

        assert "image" in str(exc_info.value)

    def test_empty_image_raises_error(self) -> None:
        """Empty image string raises ValidationError due to min_length=1."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest(image="")

        assert "image" in str(exc_info.value).lower()

    def test_extra_field_raises_error(self) -> None:
        """Extra field raises ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest(image="data", unknown_field="value")  # type: ignore[call-arg]

        assert "extra" in str(exc_info.value).lower()


@pytest.mark.unit
class TestReferenceFeatures:
    """Tests for reference features schema."""

    def test_valid_reference_features(self) -> None:
        """Valid reference features."""
        features = ReferenceFeatures(
            keypoints=[
                KeypointData(x=10.0, y=20.0, size=5.0, angle=0.0),
                KeypointData(x=30.0, y=40.0, size=6.0, angle=90.0),
            ],
            descriptors="base64encodeddescriptors",
        )

        assert len(features.keypoints) == 2
        assert features.descriptors == "base64encodeddescriptors"


@pytest.mark.unit
class TestMatchRequest:
    """Tests for match request schema."""

    def test_valid_request_with_reference_image(self) -> None:
        """Valid request with reference_image."""
        request = MatchRequest(
            query_image="base64queryimage",
            reference_image="base64refimage",
        )

        assert request.query_image == "base64queryimage"
        assert request.reference_image == "base64refimage"
        assert request.reference_features is None

    def test_valid_request_with_reference_features(self) -> None:
        """Valid request with pre-extracted reference_features."""
        features = ReferenceFeatures(
            keypoints=[KeypointData(x=10.0, y=20.0, size=5.0, angle=0.0)],
            descriptors="base64desc",
        )
        request = MatchRequest(
            query_image="base64queryimage",
            reference_features=features,
        )

        assert request.reference_features is not None
        assert len(request.reference_features.keypoints) == 1

    def test_valid_request_with_ids(self) -> None:
        """Valid request with query and reference IDs."""
        request = MatchRequest(
            query_image="base64queryimage",
            reference_image="base64refimage",
            query_id="query_001",
            reference_id="ref_001",
        )

        assert request.query_id == "query_001"
        assert request.reference_id == "ref_001"

    def test_missing_query_image_raises_error(self) -> None:
        """Missing query_image raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MatchRequest(reference_image="base64refimage")  # type: ignore[call-arg]

        assert "query_image" in str(exc_info.value)

    def test_empty_query_image_raises_error(self) -> None:
        """Empty query_image string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MatchRequest(query_image="", reference_image="base64refimage")

        assert "query_image" in str(exc_info.value).lower()


@pytest.mark.unit
class TestReferenceInput:
    """Tests for reference input schema."""

    def test_valid_reference_input_with_image(self) -> None:
        """Valid reference input with image."""
        ref = ReferenceInput(
            reference_id="ref_001",
            reference_image="base64image",
        )

        assert ref.reference_id == "ref_001"
        assert ref.reference_image == "base64image"

    def test_valid_reference_input_with_features(self) -> None:
        """Valid reference input with pre-extracted features."""
        features = ReferenceFeatures(
            keypoints=[KeypointData(x=10.0, y=20.0, size=5.0, angle=0.0)],
            descriptors="base64desc",
        )
        ref = ReferenceInput(
            reference_id="ref_001",
            reference_features=features,
        )

        assert ref.reference_features is not None


@pytest.mark.unit
class TestBatchMatchRequest:
    """Tests for batch match request schema."""

    def test_valid_batch_request(self) -> None:
        """Valid batch match request."""
        request = BatchMatchRequest(
            query_image="base64queryimage",
            references=[
                ReferenceInput(reference_id="ref_001", reference_image="img1"),
                ReferenceInput(reference_id="ref_002", reference_image="img2"),
            ],
        )

        assert len(request.references) == 2
        assert request.query_id is None

    def test_valid_batch_request_with_query_id(self) -> None:
        """Valid batch request with query_id."""
        request = BatchMatchRequest(
            query_image="base64queryimage",
            references=[ReferenceInput(reference_id="ref_001", reference_image="img1")],
            query_id="query_001",
        )

        assert request.query_id == "query_001"


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


@pytest.mark.unit
class TestAlgorithmInfo:
    """Tests for algorithm info schema."""

    def test_valid_algorithm_info(self) -> None:
        """Valid algorithm info."""
        info = AlgorithmInfo(
            feature_detector="ORB",
            max_features=1000,
            matcher="BFMatcher",
            matcher_norm="HAMMING",
            ratio_threshold=0.75,
            verification="RANSAC",
            ransac_reproj_threshold=5.0,
            min_inliers=10,
        )

        assert info.feature_detector == "ORB"
        assert info.max_features == 1000
        assert info.ratio_threshold == 0.75


@pytest.mark.unit
class TestInfoResponse:
    """Tests for info response schema."""

    def test_valid_info_response(self) -> None:
        """Valid info response with nested algorithm info."""
        response = InfoResponse(
            service="geometric",
            version="0.1.0",
            algorithm=AlgorithmInfo(
                feature_detector="ORB",
                max_features=1000,
                matcher="BFMatcher",
                matcher_norm="HAMMING",
                ratio_threshold=0.75,
                verification="RANSAC",
                ransac_reproj_threshold=5.0,
                min_inliers=10,
            ),
        )

        assert response.service == "geometric"
        assert response.algorithm.feature_detector == "ORB"


@pytest.mark.unit
class TestExtractResponse:
    """Tests for extract response schema."""

    def test_valid_extract_response(self) -> None:
        """Valid extract response."""
        response = ExtractResponse(
            image_id="test_001",
            num_features=100,
            keypoints=[KeypointData(x=10.0, y=20.0, size=5.0, angle=0.0)],
            descriptors="base64descriptors",
            image_size=ImageSize(width=640, height=480),
            processing_time_ms=45.2,
        )

        assert response.num_features == 100
        assert len(response.keypoints) == 1
        assert response.image_size.width == 640

    def test_extract_response_with_none_image_id(self) -> None:
        """Extract response with null image_id."""
        response = ExtractResponse(
            image_id=None,
            num_features=50,
            keypoints=[],
            descriptors="",
            image_size=ImageSize(width=320, height=240),
            processing_time_ms=10.0,
        )

        assert response.image_id is None


@pytest.mark.unit
class TestMatchResponse:
    """Tests for match response schema."""

    def test_valid_match_response_with_match(self) -> None:
        """Valid match response for a successful match."""
        response = MatchResponse(
            is_match=True,
            confidence=0.85,
            inliers=25,
            total_matches=50,
            inlier_ratio=0.5,
            query_features=100,
            reference_features=120,
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            query_id="query_001",
            reference_id="ref_001",
            processing_time_ms=150.5,
        )

        assert response.is_match is True
        assert response.confidence == 0.85
        assert response.inliers == 25
        assert response.homography is not None
        assert len(response.homography) == 3

    def test_valid_match_response_no_match(self) -> None:
        """Valid match response for no match found."""
        response = MatchResponse(
            is_match=False,
            confidence=0.1,
            inliers=3,
            total_matches=20,
            inlier_ratio=0.15,
            query_features=80,
            reference_features=90,
            homography=None,
            processing_time_ms=100.0,
        )

        assert response.is_match is False
        assert response.homography is None
        assert response.query_id is None
        assert response.reference_id is None

    def test_match_response_optional_fields_default_none(self) -> None:
        """Match response optional fields default to None."""
        response = MatchResponse(
            is_match=False,
            confidence=0.0,
            inliers=0,
            total_matches=0,
            inlier_ratio=0.0,
            query_features=50,
            reference_features=60,
            processing_time_ms=50.0,
        )

        assert response.homography is None
        assert response.query_id is None
        assert response.reference_id is None


@pytest.mark.unit
class TestBatchMatchResult:
    """Tests for batch match result schema."""

    def test_valid_batch_result(self) -> None:
        """Valid batch match result."""
        result = BatchMatchResult(
            reference_id="ref_001",
            is_match=True,
            confidence=0.9,
            inliers=30,
            inlier_ratio=0.6,
        )

        assert result.reference_id == "ref_001"
        assert result.is_match is True


@pytest.mark.unit
class TestBestMatch:
    """Tests for best match schema."""

    def test_valid_best_match(self) -> None:
        """Valid best match."""
        best = BestMatch(
            reference_id="ref_001",
            confidence=0.95,
        )

        assert best.reference_id == "ref_001"
        assert best.confidence == 0.95


@pytest.mark.unit
class TestBatchMatchResponse:
    """Tests for batch match response schema."""

    def test_valid_batch_response_with_best_match(self) -> None:
        """Valid batch response with best match."""
        response = BatchMatchResponse(
            query_id="query_001",
            query_features=100,
            results=[
                BatchMatchResult(
                    reference_id="ref_001",
                    is_match=True,
                    confidence=0.9,
                    inliers=30,
                    inlier_ratio=0.6,
                ),
                BatchMatchResult(
                    reference_id="ref_002",
                    is_match=False,
                    confidence=0.2,
                    inliers=5,
                    inlier_ratio=0.1,
                ),
            ],
            best_match=BestMatch(reference_id="ref_001", confidence=0.9),
            processing_time_ms=500.0,
        )

        assert response.query_id == "query_001"
        assert len(response.results) == 2
        assert response.best_match is not None
        assert response.best_match.reference_id == "ref_001"

    def test_valid_batch_response_no_best_match(self) -> None:
        """Valid batch response with no best match."""
        response = BatchMatchResponse(
            query_id=None,
            query_features=50,
            results=[
                BatchMatchResult(
                    reference_id="ref_001",
                    is_match=False,
                    confidence=0.1,
                    inliers=2,
                    inlier_ratio=0.05,
                ),
            ],
            best_match=None,
            processing_time_ms=200.0,
        )

        assert response.best_match is None


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
