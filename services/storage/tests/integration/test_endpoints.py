"""Integration tests for all endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestHealthEndpoint:
    """Integration tests for /health."""

    def test_health_returns_healthy(self, client) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.integration
class TestInfoEndpoint:
    """Integration tests for /info."""

    def test_info_returns_storage_config(self, client) -> None:
        """Info endpoint returns storage configuration."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "storage"
        assert data["storage"]["content_type"] == "image/jpeg"
        assert data["storage"]["object_count"] == 0


@pytest.mark.integration
class TestPutAndGetObject:
    """Integration tests for PUT and GET /objects/{id}."""

    def test_put_and_get_roundtrip(self, client) -> None:
        """Store and retrieve an object."""
        put_response = client.put("/objects/test-obj", content=b"hello world")
        assert put_response.status_code == 204

        get_response = client.get("/objects/test-obj")
        assert get_response.status_code == 200
        assert get_response.content == b"hello world"
        assert get_response.headers["content-type"] == "image/jpeg"

    def test_put_overwrites_existing(self, client) -> None:
        """Putting to an existing key overwrites."""
        client.put("/objects/test-obj", content=b"original")
        client.put("/objects/test-obj", content=b"updated")

        response = client.get("/objects/test-obj")
        assert response.content == b"updated"

    def test_get_missing_returns_404(self, client) -> None:
        """Getting non-existent object returns 404."""
        response = client.get("/objects/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    def test_put_binary_data(self, client) -> None:
        """Handles arbitrary binary data."""
        binary_data = bytes(range(256))
        client.put("/objects/binary", content=binary_data)

        response = client.get("/objects/binary")
        assert response.content == binary_data


@pytest.mark.integration
class TestDeleteObject:
    """Integration tests for DELETE /objects/{id}."""

    def test_delete_existing_returns_204(self, client) -> None:
        """Deleting existing object returns 204."""
        client.put("/objects/to-delete", content=b"data")
        response = client.delete("/objects/to-delete")
        assert response.status_code == 204

        get_response = client.get("/objects/to-delete")
        assert get_response.status_code == 404

    def test_delete_missing_returns_404(self, client) -> None:
        """Deleting non-existent object returns 404."""
        response = client.delete("/objects/nonexistent")
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"


@pytest.mark.integration
class TestDeleteAll:
    """Integration tests for DELETE /objects."""

    def test_delete_all_returns_count(self, client) -> None:
        """Delete all returns count of deleted objects."""
        client.put("/objects/obj1", content=b"data1")
        client.put("/objects/obj2", content=b"data2")
        client.put("/objects/obj3", content=b"data3")

        response = client.delete("/objects")
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 3

    def test_delete_all_empty_returns_zero(self, client) -> None:
        """Delete all on empty store returns 0."""
        response = client.delete("/objects")
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 0

    def test_info_reflects_count_after_operations(self, client) -> None:
        """Info endpoint object_count reflects current state."""
        client.put("/objects/obj1", content=b"data1")
        client.put("/objects/obj2", content=b"data2")

        info = client.get("/info").json()
        assert info["storage"]["object_count"] == 2

        client.delete("/objects/obj1")

        info = client.get("/info").json()
        assert info["storage"]["object_count"] == 1


@pytest.mark.integration
class TestIdValidation:
    """Integration tests for object ID validation."""

    def test_valid_ids(self, client) -> None:
        """Valid IDs are accepted."""
        for obj_id in ["simple", "with-hyphen", "with_underscore", "ABC123"]:
            response = client.put(f"/objects/{obj_id}", content=b"data")
            assert response.status_code == 204, f"ID '{obj_id}' should be valid"

    def test_invalid_id_rejected(self, client) -> None:
        """Invalid IDs are rejected."""
        response = client.put("/objects/has.dot", content=b"data")
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_id"
