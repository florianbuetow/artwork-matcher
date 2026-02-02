"""Tests for search endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Generator


def create_normalized_embedding(dimension: int) -> list[float]:
    """Create a normalized random embedding."""
    arr = np.random.randn(dimension).astype(np.float32)
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()


@pytest.fixture
def mock_search_results() -> list[MagicMock]:
    """Create mock search results."""
    from search_service.services.faiss_index import SearchResult

    return [
        SearchResult(
            object_id="obj_001",
            score=0.95,
            rank=1,
            metadata={"name": "Object 1"},
        ),
    ]


@pytest.fixture
def mock_faiss_index(mock_search_results: list[MagicMock]) -> MagicMock:
    """Create a mock FAISS index."""
    faiss_index = MagicMock()
    faiss_index.dimension = 768
    faiss_index.count = 20
    faiss_index.is_empty = False
    faiss_index.search.return_value = mock_search_results
    return faiss_index


@pytest.fixture
def mock_state_with_index(mock_faiss_index: MagicMock) -> MagicMock:
    """Create a mock app state with a working FAISS index."""
    state = MagicMock()
    state.uptime_seconds = 123.45
    state.uptime_formatted = "2m 3s"
    state.index_loaded = True
    state.index_count = 20
    state.faiss_index = mock_faiss_index
    return state


@pytest.fixture
def mock_state_no_index() -> MagicMock:
    """Create a mock app state without a loaded index."""
    state = MagicMock()
    state.uptime_seconds = 123.45
    state.uptime_formatted = "2m 3s"
    state.index_loaded = False
    state.index_count = 0
    state.faiss_index = None
    return state


@pytest.fixture
def mock_state_empty_index() -> MagicMock:
    """Create a mock app state with an empty FAISS index."""
    faiss_index = MagicMock()
    faiss_index.dimension = 768
    faiss_index.count = 0
    faiss_index.is_empty = True

    state = MagicMock()
    state.uptime_seconds = 123.45
    state.uptime_formatted = "2m 3s"
    state.index_loaded = True
    state.index_count = 0
    state.faiss_index = faiss_index
    return state


@pytest.fixture
def client(
    mock_settings: MagicMock,
    mock_state_with_index: MagicMock,
) -> Generator[TestClient, None, None]:
    """Create test client with mocked dependencies."""
    faiss_index = mock_state_with_index.faiss_index
    with (
        patch("search_service.config.get_settings", return_value=mock_settings),
        patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
        patch("search_service.core.lifespan.init_app_state", return_value=mock_state_with_index),
        patch("search_service.core.lifespan.create_faiss_index", return_value=faiss_index),
        patch("search_service.core.lifespan.try_load_index", return_value=False),
        patch("search_service.core.lifespan.setup_logging"),
        patch("search_service.core.lifespan.get_logger", return_value=MagicMock()),
        patch("search_service.core.state.get_app_state", return_value=mock_state_with_index),
        patch("search_service.routers.search.get_settings", return_value=mock_settings),
        patch("search_service.routers.search.get_app_state", return_value=mock_state_with_index),
    ):
        from search_service.app import create_app

        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


class TestSearchEndpoint:
    """Tests for POST /search endpoint."""

    def test_search_returns_results(
        self,
        client: TestClient,
    ) -> None:
        """Search endpoint returns ranked results."""
        embedding = create_normalized_embedding(768)
        response = client.post("/search", json={"embedding": embedding, "k": 5})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert "processing_time_ms" in data
        assert data["count"] == 1
        assert data["results"][0]["object_id"] == "obj_001"

    def test_search_uses_defaults(
        self,
        client: TestClient,
        mock_settings: MagicMock,
        mock_faiss_index: MagicMock,
    ) -> None:
        """Search uses config defaults when k and threshold not provided."""
        mock_faiss_index.search.return_value = []

        embedding = create_normalized_embedding(768)
        response = client.post("/search", json={"embedding": embedding})

        assert response.status_code == 200

        # Verify search was called with defaults
        mock_faiss_index.search.assert_called_once()
        call_kwargs = mock_faiss_index.search.call_args[1]
        assert call_kwargs["k"] == mock_settings.search.default_k
        assert call_kwargs["threshold"] == mock_settings.search.default_threshold

    def test_search_empty_index_returns_error(
        self,
        mock_settings: MagicMock,
        mock_state_empty_index: MagicMock,
    ) -> None:
        """Search on empty index returns appropriate error."""
        state = mock_state_empty_index
        faiss_index = state.faiss_index
        with (
            patch("search_service.config.get_settings", return_value=mock_settings),
            patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
            patch("search_service.core.lifespan.init_app_state", return_value=state),
            patch("search_service.core.lifespan.create_faiss_index", return_value=faiss_index),
            patch("search_service.core.lifespan.try_load_index", return_value=False),
            patch("search_service.core.lifespan.setup_logging"),
            patch("search_service.core.lifespan.get_logger", return_value=MagicMock()),
            patch("search_service.core.state.get_app_state", return_value=state),
            patch("search_service.routers.search.get_settings", return_value=mock_settings),
            patch("search_service.routers.search.get_app_state", return_value=state),
        ):
            from search_service.app import create_app

            app = create_app()
            test_client = TestClient(app, raise_server_exceptions=False)

            embedding = create_normalized_embedding(768)
            response = test_client.post("/search", json={"embedding": embedding})

            assert response.status_code == 422
            data = response.json()
            assert data["error"] == "index_empty"

    def test_search_dimension_mismatch_returns_error(
        self,
        mock_settings: MagicMock,
        mock_faiss_index: MagicMock,
        mock_state_with_index: MagicMock,
    ) -> None:
        """Search with wrong dimension returns error."""
        from search_service.services.faiss_index import DimensionMismatchError

        mock_faiss_index.search.side_effect = DimensionMismatchError(expected=768, received=512)
        state = mock_state_with_index
        faiss_index = state.faiss_index

        with (
            patch("search_service.config.get_settings", return_value=mock_settings),
            patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
            patch("search_service.core.lifespan.init_app_state", return_value=state),
            patch("search_service.core.lifespan.create_faiss_index", return_value=faiss_index),
            patch("search_service.core.lifespan.try_load_index", return_value=False),
            patch("search_service.core.lifespan.setup_logging"),
            patch("search_service.core.lifespan.get_logger", return_value=MagicMock()),
            patch("search_service.core.state.get_app_state", return_value=state),
            patch("search_service.routers.search.get_settings", return_value=mock_settings),
            patch("search_service.routers.search.get_app_state", return_value=state),
        ):
            from search_service.app import create_app

            app = create_app()
            test_client = TestClient(app, raise_server_exceptions=False)

            embedding = create_normalized_embedding(512)
            response = test_client.post("/search", json={"embedding": embedding})

            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "dimension_mismatch"
            assert data["details"]["expected"] == 768
            assert data["details"]["received"] == 512

    def test_search_index_not_loaded_returns_error(
        self,
        mock_settings: MagicMock,
        mock_state_no_index: MagicMock,
    ) -> None:
        """Search when index not loaded returns error."""
        with (
            patch("search_service.config.get_settings", return_value=mock_settings),
            patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
            patch("search_service.core.lifespan.init_app_state", return_value=mock_state_no_index),
            patch("search_service.core.lifespan.create_faiss_index", return_value=MagicMock()),
            patch("search_service.core.lifespan.try_load_index", return_value=False),
            patch("search_service.core.lifespan.setup_logging"),
            patch("search_service.core.lifespan.get_logger", return_value=MagicMock()),
            patch("search_service.core.state.get_app_state", return_value=mock_state_no_index),
            patch("search_service.routers.search.get_settings", return_value=mock_settings),
            patch("search_service.routers.search.get_app_state", return_value=mock_state_no_index),
        ):
            from search_service.app import create_app

            app = create_app()
            test_client = TestClient(app, raise_server_exceptions=False)

            embedding = create_normalized_embedding(768)
            response = test_client.post("/search", json={"embedding": embedding})

            assert response.status_code == 503
            data = response.json()
            assert data["error"] == "index_not_loaded"
