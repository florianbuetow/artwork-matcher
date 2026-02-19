"""Unit tests for info endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_info_returns_200(self) -> None:
        """Info endpoint returns 200 OK."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            settings.storage.path = "./data/objects"
            settings.storage.content_type = "application/octet-stream"
            mock_settings.return_value = settings

            state = MagicMock()
            state.blob_store.count.return_value = 5
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")

            assert response.status_code == 200

    def test_info_returns_storage_config(self) -> None:
        """Info endpoint returns storage configuration."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            settings.storage.path = "./data/objects"
            settings.storage.content_type = "application/octet-stream"
            mock_settings.return_value = settings

            state = MagicMock()
            state.blob_store.count.return_value = 5
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/info")
            data = response.json()

            assert data["service"] == "storage"
            assert data["storage"]["path"] == "./data/objects"
            assert data["storage"]["content_type"] == "application/octet-stream"
            assert data["storage"]["object_count"] == 5
