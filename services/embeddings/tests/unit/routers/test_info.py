"""
Unit tests for info endpoint.

Tests the /info endpoint with mocked dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from embeddings_service.app import create_app


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_info_returns_200(self, mock_settings: MagicMock, mock_app_state: MagicMock) -> None:
        """Info endpoint returns 200 OK."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")

            assert response.status_code == 200

    def test_info_includes_service_name(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Info endpoint includes service name."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["service"] == "embeddings"

    def test_info_includes_version(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Info endpoint includes version."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["version"] == "0.1.0"

    def test_info_includes_model_config(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Info endpoint includes model configuration."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert "model" in data
            model = data["model"]
            assert "name" in model
            assert "embedding_dimension" in model
            assert "device" in model

    def test_info_model_name_is_dinov2(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Model name contains dinov2."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert "dinov2" in data["model"]["name"].lower()

    def test_info_embedding_dimension_is_768(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Embedding dimension is 768 for dinov2-base."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["model"]["embedding_dimension"] == 768

    def test_info_includes_preprocessing_config(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Info endpoint includes preprocessing configuration."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert "preprocessing" in data
            preprocessing = data["preprocessing"]
            assert "image_size" in preprocessing
            assert "normalize" in preprocessing

    def test_info_image_size_is_518(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Image size is 518 (DINOv2 native)."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["preprocessing"]["image_size"] == 518

    def test_info_normalize_is_true(
        self, mock_settings: MagicMock, mock_app_state: MagicMock
    ) -> None:
        """Normalize is enabled."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.info.get_app_state") as mock_state,
            patch("embeddings_service.routers.info.get_settings") as settings_mock,
        ):
            mock_state.return_value = mock_app_state
            settings_mock.return_value = mock_settings
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["preprocessing"]["normalize"] is True
