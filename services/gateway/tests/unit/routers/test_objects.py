"""
Unit tests for objects endpoints.

Tests the /objects, /objects/{id}, and /objects/{id}/image endpoints
with mocked metadata and storage client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture
def mock_metadata() -> dict[str, dict[str, str | None]]:
    """Create mock metadata for testing."""
    return {
        "obj_001": {
            "object_id": "obj_001",
            "name": "Water Lilies",
            "artist": "Claude Monet",
            "year": "1906",
            "description": "Part of a series of paintings",
            "location": "Gallery 3",
        },
        "obj_002": {
            "object_id": "obj_002",
            "name": "Starry Night",
            "artist": "Vincent van Gogh",
            "year": "1889",
            "description": None,
            "location": None,
        },
    }


@pytest.mark.unit
class TestListObjects:
    """Tests for GET /objects."""

    def test_list_objects_returns_200(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """List objects returns 200 OK."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects")

        assert response.status_code == 200

    def test_list_objects_returns_correct_count(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """List objects returns correct count."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects")
            data = response.json()

        assert data["count"] == 2

    def test_list_objects_returns_objects(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """List objects returns objects array."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects")
            data = response.json()

        assert "objects" in data
        assert len(data["objects"]) == 2

    def test_list_objects_sorted_by_id(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """List objects are sorted by object_id."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects")
            data = response.json()

        object_ids = [obj["object_id"] for obj in data["objects"]]
        assert object_ids == sorted(object_ids)

    def test_list_objects_includes_metadata(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """List objects includes basic metadata."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects")
            data = response.json()

        first_obj = data["objects"][0]
        assert "object_id" in first_obj
        assert "name" in first_obj
        assert "artist" in first_obj
        assert "year" in first_obj

    def test_list_objects_empty_database(self, test_client: TestClient) -> None:
        """List objects handles empty database."""
        with patch("gateway.routers.objects.load_metadata", return_value={}):
            response = test_client.get("/objects")
            data = response.json()

        assert data["count"] == 0
        assert data["objects"] == []


@pytest.mark.unit
class TestGetObject:
    """Tests for GET /objects/{id}."""

    def test_get_object_returns_200(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Get object returns 200 OK for existing object."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects/obj_001")

        assert response.status_code == 200

    def test_get_object_returns_details(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Get object returns full details."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects/obj_001")
            data = response.json()

        assert data["object_id"] == "obj_001"
        assert data["name"] == "Water Lilies"
        assert data["artist"] == "Claude Monet"
        assert data["year"] == "1906"
        assert data["description"] == "Part of a series of paintings"
        assert data["location"] == "Gallery 3"

    def test_get_object_includes_image_url(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Get object always includes image_url."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects/obj_001")
            data = response.json()

        assert data["image_url"] == "/objects/obj_001/image"

    def test_get_object_not_found(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Get object returns 404 for non-existent object."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects/nonexistent")

        assert response.status_code == 404

    def test_get_object_not_found_error_details(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Get object 404 response includes error details."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects/nonexistent")
            data = response.json()

        assert data["detail"]["error"] == "not_found"
        assert "nonexistent" in data["detail"]["message"]

    def test_get_object_handles_null_fields(
        self,
        test_client: TestClient,
        mock_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Get object handles null optional fields."""
        with patch("gateway.routers.objects.load_metadata", return_value=mock_metadata):
            response = test_client.get("/objects/obj_002")
            data = response.json()

        assert data["object_id"] == "obj_002"
        assert data["name"] == "Starry Night"
        assert data["description"] is None
        assert data["location"] is None


@pytest.mark.unit
class TestGetObjectImage:
    """Tests for GET /objects/{id}/image."""

    def test_get_image_returns_200(
        self,
        test_client: TestClient,
    ) -> None:
        """Get image returns 200 OK when storage has the image."""
        response = test_client.get("/objects/obj_001/image")

        assert response.status_code == 200

    def test_get_image_returns_jpeg_content_type(
        self,
        test_client: TestClient,
    ) -> None:
        """Get image returns JPEG content type."""
        response = test_client.get("/objects/obj_001/image")

        assert response.headers["content-type"] == "image/jpeg"

    def test_get_image_returns_image_bytes(
        self,
        test_client: TestClient,
    ) -> None:
        """Get image returns the raw bytes from storage."""
        response = test_client.get("/objects/obj_001/image")

        assert response.content == b"fake jpeg data"

    def test_get_image_not_found(
        self,
        test_client: TestClient,
        mock_app_state: AsyncMock,
    ) -> None:
        """Get image returns 404 when storage returns None."""
        mock_app_state.storage_client.get_image_bytes.return_value = None

        response = test_client.get("/objects/obj_001/image")

        assert response.status_code == 404

    def test_get_image_not_found_error_details(
        self,
        test_client: TestClient,
        mock_app_state: AsyncMock,
    ) -> None:
        """Get image 404 response includes error details."""
        mock_app_state.storage_client.get_image_bytes.return_value = None

        response = test_client.get("/objects/obj_001/image")
        data = response.json()

        assert data["detail"]["error"] == "image_not_found"
        assert "obj_001" in data["detail"]["message"]
