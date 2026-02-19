"""Unit tests for objects endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestPutObject:
    """Tests for PUT /objects/{object_id}."""

    def test_put_returns_204(self) -> None:
        """Storing an object returns 204 No Content."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.put("/objects/test-id", content=b"hello")

            assert response.status_code == 204
            state.blob_store.put.assert_called_once_with("test-id", b"hello")

    def test_put_invalid_id_returns_400(self) -> None:
        """Invalid object ID returns 400."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.put("/objects/has.dot", content=b"data")

            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "invalid_id"


@pytest.mark.unit
class TestGetObject:
    """Tests for GET /objects/{object_id}."""

    def test_get_existing_returns_200(self) -> None:
        """Getting an existing object returns 200 with bytes."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
            patch("storage_service.routers.objects.get_settings") as mock_obj_settings,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            settings.storage.content_type = "image/jpeg"
            mock_settings.return_value = settings
            mock_obj_settings.return_value = settings

            state = MagicMock()
            state.blob_store.get.return_value = b"image-data"
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/objects/test-id")

            assert response.status_code == 200
            assert response.content == b"image-data"
            assert response.headers["content-type"] == "image/jpeg"

    def test_get_missing_returns_404(self) -> None:
        """Getting a non-existent object returns 404."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
            patch("storage_service.routers.objects.get_settings") as mock_obj_settings,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            settings.storage.content_type = "image/jpeg"
            mock_settings.return_value = settings
            mock_obj_settings.return_value = settings

            state = MagicMock()
            state.blob_store.get.return_value = None
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/objects/nonexistent")

            assert response.status_code == 404
            data = response.json()
            assert data["error"] == "not_found"


@pytest.mark.unit
class TestDeleteObject:
    """Tests for DELETE /objects/{object_id}."""

    def test_delete_existing_returns_204(self) -> None:
        """Deleting an existing object returns 204."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.blob_store.delete.return_value = True
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.delete("/objects/test-id")

            assert response.status_code == 204

    def test_delete_missing_returns_404(self) -> None:
        """Deleting a non-existent object returns 404."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.blob_store.delete.return_value = False
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.delete("/objects/nonexistent")

            assert response.status_code == 404
            data = response.json()
            assert data["error"] == "not_found"


@pytest.mark.unit
class TestDeleteAll:
    """Tests for DELETE /objects."""

    def test_delete_all_returns_count(self) -> None:
        """Deleting all objects returns the count."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            state.blob_store.delete_all.return_value = 3
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.delete("/objects")

            assert response.status_code == 200
            data = response.json()
            assert data["deleted_count"] == 3


@pytest.mark.unit
class TestIdValidation:
    """Tests for object ID validation."""

    @pytest.mark.parametrize(
        "valid_id",
        ["simple", "with-hyphen", "with_underscore", "MiXeD123", "abc123"],
    )
    def test_valid_ids_accepted(self, valid_id: str) -> None:
        """Valid IDs are accepted."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state") as mock_state,
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            state = MagicMock()
            mock_state.return_value = state

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.put(f"/objects/{valid_id}", content=b"data")

            assert response.status_code == 204

    @pytest.mark.parametrize(
        "invalid_id",
        ["has.dot", "has@symbol", "has+plus"],
    )
    def test_invalid_ids_rejected(self, invalid_id: str) -> None:
        """Invalid IDs are rejected with 400."""
        with (
            patch("storage_service.config.get_settings") as mock_settings,
            patch("storage_service.app.lifespan"),
            patch("storage_service.routers.health.get_app_state"),
            patch("storage_service.routers.info.get_app_state"),
            patch("storage_service.routers.objects.get_app_state"),
        ):
            settings = MagicMock()
            settings.service.name = "storage"
            settings.service.version = "0.1.0"
            mock_settings.return_value = settings

            from storage_service.app import create_app  # noqa: PLC0415

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.put(f"/objects/{invalid_id}", content=b"data")

            assert response.status_code == 400
