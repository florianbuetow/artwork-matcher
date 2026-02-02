"""Tests for index management endpoints."""

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
def mock_faiss_index() -> MagicMock:
    """Create a mock FAISS index."""
    faiss_index = MagicMock()
    faiss_index.dimension = 768
    faiss_index.count = 5
    faiss_index.is_empty = False
    faiss_index.add.return_value = 0
    faiss_index.save.return_value = 1024
    faiss_index.clear.return_value = 5
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
def client_with_index(
    mock_settings: MagicMock,
    mock_state_with_index: MagicMock,
) -> Generator[TestClient, None, None]:
    """Create test client with mocked dependencies and loaded index."""
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
        patch("search_service.routers.index.get_settings", return_value=mock_settings),
        patch("search_service.routers.index.get_app_state", return_value=mock_state_with_index),
        patch("search_service.routers.index.get_logger", return_value=MagicMock()),
    ):
        from search_service.app import create_app

        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_no_index(
    mock_settings: MagicMock,
    mock_state_no_index: MagicMock,
) -> Generator[TestClient, None, None]:
    """Create test client without loaded index."""
    with (
        patch("search_service.config.get_settings", return_value=mock_settings),
        patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
        patch("search_service.core.lifespan.init_app_state", return_value=mock_state_no_index),
        patch("search_service.core.lifespan.create_faiss_index", return_value=MagicMock()),
        patch("search_service.core.lifespan.try_load_index", return_value=False),
        patch("search_service.core.lifespan.setup_logging"),
        patch("search_service.core.lifespan.get_logger", return_value=MagicMock()),
        patch("search_service.core.state.get_app_state", return_value=mock_state_no_index),
        patch("search_service.routers.index.get_settings", return_value=mock_settings),
        patch("search_service.routers.index.get_app_state", return_value=mock_state_no_index),
        patch("search_service.routers.index.get_logger", return_value=MagicMock()),
    ):
        from search_service.app import create_app

        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


@pytest.mark.unit
class TestAddEndpoint:
    """Tests for POST /add endpoint."""

    def test_add_returns_success(
        self,
        client_with_index: TestClient,
        mock_faiss_index: MagicMock,
    ) -> None:
        """Add endpoint adds embedding successfully."""
        mock_faiss_index.add.return_value = 0
        mock_faiss_index.count = 1

        embedding = create_normalized_embedding(768)
        response = client_with_index.post(
            "/add",
            json={
                "object_id": "test_001",
                "embedding": embedding,
                "metadata": {"name": "Test Object"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["object_id"] == "test_001"
        assert data["index_position"] == 0
        assert data["index_count"] == 1

    def test_add_dimension_mismatch_returns_error(
        self,
        client_with_index: TestClient,
        mock_faiss_index: MagicMock,
    ) -> None:
        """Add with wrong dimension returns error."""
        from search_service.services.faiss_index import DimensionMismatchError

        mock_faiss_index.add.side_effect = DimensionMismatchError(expected=768, received=512)

        embedding = create_normalized_embedding(512)
        response = client_with_index.post(
            "/add",
            json={"object_id": "test_001", "embedding": embedding},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "dimension_mismatch"

    def test_add_index_not_loaded_returns_error(
        self,
        client_no_index: TestClient,
    ) -> None:
        """Add when index not loaded returns error."""
        embedding = create_normalized_embedding(768)
        response = client_no_index.post(
            "/add",
            json={"object_id": "test_001", "embedding": embedding},
        )

        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "index_not_loaded"


@pytest.mark.unit
class TestSaveEndpoint:
    """Tests for POST /index/save endpoint."""

    def test_save_returns_success(
        self,
        client_with_index: TestClient,
        mock_settings: MagicMock,
        mock_faiss_index: MagicMock,
    ) -> None:
        """Save endpoint saves index successfully."""
        mock_faiss_index.save.return_value = 1024
        mock_faiss_index.count = 5

        response = client_with_index.post("/index/save", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["index_path"] == mock_settings.index.path
        assert data["count"] == 5
        assert data["size_bytes"] == 1024

    def test_save_index_not_loaded_returns_error(
        self,
        client_no_index: TestClient,
    ) -> None:
        """Save when index not loaded returns error."""
        response = client_no_index.post("/index/save", json={})

        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "index_not_loaded"


@pytest.mark.unit
class TestLoadEndpoint:
    """Tests for POST /index/load endpoint."""

    def test_load_returns_success(
        self,
        client_with_index: TestClient,
        mock_settings: MagicMock,
        mock_faiss_index: MagicMock,
    ) -> None:
        """Load endpoint loads index successfully."""
        mock_faiss_index.count = 10
        mock_faiss_index.dimension = 768

        response = client_with_index.post("/index/load", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["index_path"] == mock_settings.index.path
        assert data["count"] == 10
        assert data["dimension"] == 768


@pytest.mark.unit
class TestClearEndpoint:
    """Tests for DELETE /index endpoint."""

    def test_clear_returns_success(
        self,
        client_with_index: TestClient,
        mock_faiss_index: MagicMock,
    ) -> None:
        """Clear endpoint clears index successfully."""
        mock_faiss_index.clear.return_value = 5
        mock_faiss_index.count = 0

        response = client_with_index.delete("/index")

        assert response.status_code == 200
        data = response.json()
        assert data["previous_count"] == 5
        assert data["current_count"] == 0

    def test_clear_index_not_loaded_returns_error(
        self,
        client_no_index: TestClient,
    ) -> None:
        """Clear when index not loaded returns error."""
        response = client_no_index.delete("/index")

        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "index_not_loaded"
