"""Tests for GeometricClient."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.clients.geometric import BatchMatchResult, GeometricResult

if TYPE_CHECKING:
    from gateway.clients.geometric import GeometricClient


@pytest.mark.unit
class TestGeometricResult:
    """Tests for GeometricResult class."""

    def test_geometric_result_parses_all_fields(self) -> None:
        """Test GeometricResult correctly parses all fields."""
        data = {
            "reference_id": "ref_001",
            "is_match": True,
            "confidence": 0.85,
            "inliers": 42,
            "inlier_ratio": 0.7,
        }

        result = GeometricResult(data)

        assert result.reference_id == "ref_001"
        assert result.is_match is True
        assert result.confidence == 0.85
        assert result.inliers == 42
        assert result.inlier_ratio == 0.7

    def test_geometric_result_missing_reference_id_raises_value_error(self) -> None:
        """Test GeometricResult raises ValueError when reference_id is missing."""
        data = {"is_match": True, "confidence": 0.8}

        with pytest.raises(ValueError) as exc_info:
            GeometricResult(data)

        assert "reference_id" in str(exc_info.value)

    def test_geometric_result_missing_is_match_raises_value_error(self) -> None:
        """Test GeometricResult raises ValueError when is_match is missing."""
        data = {"reference_id": "ref_001", "confidence": 0.8}

        with pytest.raises(ValueError) as exc_info:
            GeometricResult(data)

        assert "is_match" in str(exc_info.value)

    def test_geometric_result_missing_confidence_raises_value_error(self) -> None:
        """Test GeometricResult raises ValueError when confidence is missing."""
        data = {"reference_id": "ref_001", "is_match": True}

        with pytest.raises(ValueError) as exc_info:
            GeometricResult(data)

        assert "confidence" in str(exc_info.value)

    def test_geometric_result_handles_missing_optional_fields(self) -> None:
        """Test GeometricResult handles missing optional fields with defaults."""
        data = {
            "reference_id": "ref_001",
            "is_match": False,
            "confidence": 0.3,
        }

        result = GeometricResult(data)

        assert result.inliers == 0  # Default
        assert result.inlier_ratio == 0.0  # Default


@pytest.mark.unit
class TestBatchMatchResult:
    """Tests for BatchMatchResult class."""

    def test_batch_match_result_parses_all_fields(self) -> None:
        """Test BatchMatchResult correctly parses all fields."""
        data = {
            "query_id": "query_001",
            "query_features": 500,
            "results": [
                {
                    "reference_id": "ref_001",
                    "is_match": True,
                    "confidence": 0.9,
                    "inliers": 50,
                    "inlier_ratio": 0.8,
                },
                {
                    "reference_id": "ref_002",
                    "is_match": False,
                    "confidence": 0.2,
                    "inliers": 5,
                    "inlier_ratio": 0.1,
                },
            ],
            "best_match": {
                "reference_id": "ref_001",
                "is_match": True,
                "confidence": 0.9,
                "inliers": 50,
                "inlier_ratio": 0.8,
            },
            "processing_time_ms": 123.45,
        }

        result = BatchMatchResult(data)

        assert result.query_id == "query_001"
        assert result.query_features == 500
        assert len(result.results) == 2
        assert result.results[0].reference_id == "ref_001"
        assert result.best_match is not None
        assert result.best_match.reference_id == "ref_001"
        assert result.processing_time_ms == 123.45

    def test_batch_match_result_handles_empty_results(self) -> None:
        """Test BatchMatchResult handles empty results list."""
        data = {
            "query_id": "query_001",
            "query_features": 500,
            "results": [],
            "processing_time_ms": 50.0,
        }

        result = BatchMatchResult(data)

        assert result.results == []
        assert result.best_match is None

    def test_batch_match_result_handles_missing_optional_fields(self) -> None:
        """Test BatchMatchResult handles missing optional fields."""
        data = {}  # All fields optional

        result = BatchMatchResult(data)

        assert result.query_id is None
        assert result.query_features == 0
        assert result.results == []
        assert result.best_match is None
        assert result.processing_time_ms == 0.0


@pytest.mark.unit
class TestGeometricClientMatch:
    """Tests for GeometricClient.match method."""

    async def test_match_success_returns_result(
        self,
        geometric_client: GeometricClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful geometric match returns GeometricResult."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "reference_id": "ref_001",
            "is_match": True,
            "confidence": 0.85,
            "inliers": 42,
            "inlier_ratio": 0.7,
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await geometric_client.match(
            query_image="base64_query",
            reference_image="base64_reference",
        )

        assert isinstance(result, GeometricResult)
        assert result.reference_id == "ref_001"
        assert result.is_match is True
        assert result.confidence == 0.85

    async def test_match_with_optional_ids(
        self,
        geometric_client: GeometricClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test match with optional query_id and reference_id."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "reference_id": "ref_001",
            "is_match": False,
            "confidence": 0.2,
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        await geometric_client.match(
            query_image="base64_query",
            reference_image="base64_reference",
            query_id="query_001",
            reference_id="ref_001",
        )

        # Verify the request was made with optional IDs
        call_kwargs = mock_httpx_client.request.call_args
        assert call_kwargs[1]["json"]["query_id"] == "query_001"
        assert call_kwargs[1]["json"]["reference_id"] == "ref_001"

    async def test_match_invalid_response_raises_value_error(
        self,
        geometric_client: GeometricClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test match with invalid response raises ValueError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"is_match": True}  # Missing required fields
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(ValueError) as exc_info:
            await geometric_client.match(
                query_image="base64_query",
                reference_image="base64_reference",
            )

        assert "reference_id" in str(exc_info.value)


@pytest.mark.unit
class TestGeometricClientMatchBatch:
    """Tests for GeometricClient.match_batch method."""

    async def test_match_batch_success_returns_batch_result(
        self,
        geometric_client: GeometricClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful batch match returns BatchMatchResult."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query_id": "query_001",
            "query_features": 500,
            "results": [
                {
                    "reference_id": "ref_001",
                    "is_match": True,
                    "confidence": 0.9,
                    "inliers": 50,
                    "inlier_ratio": 0.8,
                },
            ],
            "best_match": {
                "reference_id": "ref_001",
                "is_match": True,
                "confidence": 0.9,
                "inliers": 50,
                "inlier_ratio": 0.8,
            },
            "processing_time_ms": 123.45,
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await geometric_client.match_batch(
            query_image="base64_query",
            references=[
                {"reference_id": "ref_001", "reference_image": "base64_ref"},
            ],
        )

        assert isinstance(result, BatchMatchResult)
        assert result.query_id == "query_001"
        assert len(result.results) == 1
        assert result.best_match is not None

    async def test_match_batch_with_query_id(
        self,
        geometric_client: GeometricClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test batch match with optional query_id."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query_id": "query_001",
            "results": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        await geometric_client.match_batch(
            query_image="base64_query",
            references=[],
            query_id="my_query_001",
        )

        # Verify the request was made with query_id
        call_kwargs = mock_httpx_client.request.call_args
        assert call_kwargs[1]["json"]["query_id"] == "my_query_001"
