"""Tests for SearchClient."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.clients.search import SearchResult
from gateway.core.exceptions import BackendError

if TYPE_CHECKING:
    from gateway.clients.search import SearchClient


@pytest.mark.unit
class TestSearchResult:
    """Tests for SearchResult class."""

    def test_search_result_parses_all_fields(self) -> None:
        """Test SearchResult correctly parses all fields."""
        data = {
            "object_id": "obj_001",
            "score": 0.95,
            "rank": 1,
            "metadata": {"name": "Test Artwork", "artist": "Test Artist"},
        }

        result = SearchResult(data)

        assert result.object_id == "obj_001"
        assert result.score == 0.95
        assert result.rank == 1
        assert result.metadata == {"name": "Test Artwork", "artist": "Test Artist"}

    def test_search_result_missing_object_id_raises_value_error(self) -> None:
        """Test SearchResult raises ValueError when object_id is missing."""
        data = {"score": 0.95, "rank": 1}

        with pytest.raises(ValueError) as exc_info:
            SearchResult(data)

        assert "object_id" in str(exc_info.value)

    def test_search_result_missing_score_raises_value_error(self) -> None:
        """Test SearchResult raises ValueError when score is missing."""
        data = {"object_id": "obj_001", "rank": 1}

        with pytest.raises(ValueError) as exc_info:
            SearchResult(data)

        assert "score" in str(exc_info.value)

    def test_search_result_handles_missing_optional_fields(self) -> None:
        """Test SearchResult handles missing optional fields with defaults."""
        data = {"object_id": "obj_001", "score": 0.9}  # No rank or metadata

        result = SearchResult(data)

        assert result.object_id == "obj_001"
        assert result.score == 0.9
        assert result.rank == 0  # Default
        assert result.metadata == {}  # Default


@pytest.mark.unit
class TestSearchClientSearch:
    """Tests for SearchClient.search method."""

    async def test_search_success_returns_results(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful search returns list of SearchResult."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"object_id": "obj_001", "score": 0.95, "rank": 1},
                {"object_id": "obj_002", "score": 0.85, "rank": 2},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        results = await search_client.search(
            embedding=[0.1, 0.2, 0.3],
            k=5,
            threshold=0.7,
        )

        assert len(results) == 2
        assert results[0].object_id == "obj_001"
        assert results[0].score == 0.95
        assert results[1].object_id == "obj_002"

    async def test_search_missing_results_field_raises_backend_error(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that missing results field raises BackendError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}  # No results field
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(BackendError) as exc_info:
            await search_client.search(
                embedding=[0.1, 0.2, 0.3],
                k=5,
                threshold=0.7,
            )

        assert exc_info.value.error == "invalid_response"
        assert "results" in exc_info.value.message
        assert exc_info.value.status_code == 502

    async def test_search_empty_results_returns_empty_list(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test search with empty results returns empty list."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        results = await search_client.search(
            embedding=[0.1, 0.2, 0.3],
            k=5,
            threshold=0.9,
        )

        assert results == []

    async def test_search_invalid_result_raises_value_error(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test search with invalid result data raises ValueError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"score": 0.95}]  # Missing object_id
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(ValueError) as exc_info:
            await search_client.search(
                embedding=[0.1, 0.2, 0.3],
                k=5,
                threshold=0.7,
            )

        assert "object_id" in str(exc_info.value)


@pytest.mark.unit
class TestSearchClientGetIndexCount:
    """Tests for SearchClient.get_index_count method."""

    async def test_get_index_count_success(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful retrieval of index count."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "search",
            "index": {
                "count": 1500,
                "is_loaded": True,
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await search_client.get_index_count()

        assert result == 1500

    async def test_get_index_count_missing_index_info_raises_key_error(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that missing index info raises KeyError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"service": "search"}  # No index field
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(KeyError) as exc_info:
            await search_client.get_index_count()

        assert "index info" in str(exc_info.value)

    async def test_get_index_count_missing_count_raises_key_error(
        self,
        search_client: SearchClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that missing count in index info raises KeyError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "search",
            "index": {"is_loaded": True},  # No count
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(KeyError) as exc_info:
            await search_client.get_index_count()

        assert "count" in str(exc_info.value)
