"""Tests for info endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def client(mock_settings: MagicMock, mock_app_state: MagicMock) -> Iterator[TestClient]:
    """Create test client with mocked dependencies."""
    with (
        patch("search_service.config.get_settings", return_value=mock_settings),
        patch("search_service.core.state.get_app_state", return_value=mock_app_state),
        patch("search_service.core.lifespan.get_settings", return_value=mock_settings),
        patch("search_service.routers.info.get_app_state", return_value=mock_app_state),
    ):
        from search_service.app import create_app

        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info endpoint."""

    def test_info_returns_service_info(self, client: TestClient, mock_settings: MagicMock) -> None:
        """Info endpoint returns service name and version."""
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == mock_settings.service.name
        assert data["version"] == mock_settings.service.version

    def test_info_returns_index_info(
        self, client: TestClient, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Info endpoint returns index statistics."""
        response = client.get("/info")

        data = response.json()
        assert "index" in data
        assert data["index"]["type"] == mock_settings.faiss.index_type
        assert data["index"]["metric"] == mock_settings.faiss.metric
        assert data["index"]["embedding_dimension"] == mock_settings.faiss.embedding_dimension
        assert data["index"]["count"] == mock_app_state.index_count
        assert data["index"]["is_loaded"] == mock_app_state.index_loaded

    def test_info_returns_config(self, client: TestClient, mock_settings: MagicMock) -> None:
        """Info endpoint returns configuration."""
        response = client.get("/info")

        data = response.json()
        assert "config" in data
        assert data["config"]["index_path"] == mock_settings.index.path
        assert data["config"]["metadata_path"] == mock_settings.index.metadata_path
        assert data["config"]["default_k"] == mock_settings.search.default_k
