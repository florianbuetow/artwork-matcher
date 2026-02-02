"""
Integration tests for endpoint happy paths.

These tests verify that each endpoint returns correct responses
with valid data in the expected format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.factories import create_normalized_embedding

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

DIMENSION = 768  # Must match config.yaml


def add_embedding(
    client: TestClient,
    object_id: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> dict:
    """Add an embedding to the index."""
    response = client.post(
        "/add",
        json={
            "object_id": object_id,
            "embedding": embedding,
            "metadata": metadata or {},
        },
    )
    assert response.status_code == 201, f"Failed to add: {response.text}"
    return response.json()


# =============================================================================
# Health Endpoint Tests
# =============================================================================


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_uptime_seconds(self, client: TestClient) -> None:
        """Health response includes uptime_seconds field."""
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_includes_system_time(self, client: TestClient) -> None:
        """Health response includes system_time field."""
        response = client.get("/health")
        data = response.json()
        assert "system_time" in data
        assert isinstance(data["system_time"], str)
        # Basic format check: should contain date separator
        assert "-" in data["system_time"]


# =============================================================================
# Info Endpoint Tests
# =============================================================================


@pytest.mark.integration
class TestInfoEndpoint:
    """Tests for GET /info endpoint."""

    def test_info_returns_200(self, client: TestClient) -> None:
        """Info endpoint returns 200 OK."""
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_returns_service_name(self, client: TestClient) -> None:
        """Info response includes service name."""
        response = client.get("/info")
        data = response.json()
        assert "service" in data
        assert data["service"] == "search"

    def test_info_returns_version(self, client: TestClient) -> None:
        """Info response includes version string."""
        response = client.get("/info")
        data = response.json()
        assert "version" in data
        # Version should be semver format
        assert "." in data["version"]

    def test_info_returns_index_info(self, client: TestClient) -> None:
        """Info response includes index information."""
        response = client.get("/info")
        data = response.json()
        assert "index" in data
        index_info = data["index"]
        assert "type" in index_info
        assert "metric" in index_info
        assert "embedding_dimension" in index_info
        assert "count" in index_info
        assert "is_loaded" in index_info
        assert index_info["embedding_dimension"] == DIMENSION

    def test_info_returns_config_info(self, client: TestClient) -> None:
        """Info response includes configuration information."""
        response = client.get("/info")
        data = response.json()
        assert "config" in data
        config_info = data["config"]
        assert "index_path" in config_info
        assert "metadata_path" in config_info
        assert "default_k" in config_info

    def test_info_shows_correct_count_after_add(self, client: TestClient) -> None:
        """Info endpoint reflects correct count after adding embeddings."""
        # Start with empty index
        response = client.get("/info")
        initial_count = response.json()["index"]["count"]
        assert initial_count == 0

        # Add some embeddings
        for i in range(3):
            emb = create_normalized_embedding(DIMENSION, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        # Check count updated
        response = client.get("/info")
        new_count = response.json()["index"]["count"]
        assert new_count == 3


# =============================================================================
# Index Save Endpoint Tests
# =============================================================================


@pytest.mark.integration
class TestIndexSaveEndpoint:
    """Tests for POST /index/save endpoint."""

    def test_save_empty_index_succeeds(self, client: TestClient, temp_index_dir: Path) -> None:
        """Saving an empty index succeeds."""
        save_path = str(temp_index_dir / "empty.index")
        response = client.post("/index/save", json={"path": save_path})
        assert response.status_code == 200

    def test_save_populated_index_succeeds(self, client: TestClient, temp_index_dir: Path) -> None:
        """Saving a populated index succeeds."""
        # Add some data
        for i in range(5):
            emb = create_normalized_embedding(DIMENSION, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        save_path = str(temp_index_dir / "populated.index")
        response = client.post("/index/save", json={"path": save_path})
        assert response.status_code == 200

    def test_save_returns_correct_count(self, client: TestClient, temp_index_dir: Path) -> None:
        """Save response includes correct count."""
        # Add some data
        for i in range(5):
            emb = create_normalized_embedding(DIMENSION, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        save_path = str(temp_index_dir / "counted.index")
        response = client.post("/index/save", json={"path": save_path})
        data = response.json()
        assert data["count"] == 5

    def test_save_returns_size_bytes(self, client: TestClient, temp_index_dir: Path) -> None:
        """Save response includes size_bytes field."""
        emb = create_normalized_embedding(DIMENSION, seed=1)
        add_embedding(client, "obj_1", emb)

        save_path = str(temp_index_dir / "sized.index")
        response = client.post("/index/save", json={"path": save_path})
        data = response.json()
        assert "size_bytes" in data
        assert isinstance(data["size_bytes"], int)
        assert data["size_bytes"] > 0

    def test_save_creates_files(self, client: TestClient, temp_index_dir: Path) -> None:
        """Save creates index and metadata files."""
        emb = create_normalized_embedding(DIMENSION, seed=1)
        add_embedding(client, "obj_1", emb)

        save_path = temp_index_dir / "files.index"
        response = client.post("/index/save", json={"path": str(save_path)})
        assert response.status_code == 200

        # Check files exist
        assert save_path.exists(), "Index file should exist"
        metadata_path = save_path.with_suffix(".json")
        assert metadata_path.exists(), "Metadata file should exist"


# =============================================================================
# Index Load Endpoint Tests
# =============================================================================


@pytest.mark.integration
class TestIndexLoadEndpoint:
    """Tests for POST /index/load endpoint."""

    def test_load_saved_index_succeeds(self, client: TestClient, temp_index_dir: Path) -> None:
        """Loading a previously saved index succeeds."""
        # Add data and save
        for i in range(5):
            emb = create_normalized_embedding(DIMENSION, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        save_path = str(temp_index_dir / "loadable.index")
        client.post("/index/save", json={"path": save_path})

        # Clear index
        client.delete("/index")

        # Load saved index
        response = client.post("/index/load", json={"path": save_path})
        assert response.status_code == 200

    def test_load_restores_correct_count(self, client: TestClient, temp_index_dir: Path) -> None:
        """Loaded index has correct count."""
        # Add data and save
        for i in range(7):
            emb = create_normalized_embedding(DIMENSION, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        save_path = str(temp_index_dir / "counted.index")
        client.post("/index/save", json={"path": save_path})

        # Clear and reload
        client.delete("/index")
        response = client.post("/index/load", json={"path": save_path})
        data = response.json()
        assert data["count"] == 7

    def test_load_preserves_metadata(self, client: TestClient, temp_index_dir: Path) -> None:
        """Loaded index preserves metadata."""
        # Add data with metadata
        emb = create_normalized_embedding(DIMENSION, seed=42)
        add_embedding(client, "with_meta", emb, {"name": "Test Object", "score": 0.95})

        save_path = str(temp_index_dir / "with_meta.index")
        client.post("/index/save", json={"path": save_path})

        # Clear and reload
        client.delete("/index")
        client.post("/index/load", json={"path": save_path})

        # Search should return the object with its metadata
        response = client.post("/search", json={"embedding": emb, "k": 1, "threshold": 0.0})
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["object_id"] == "with_meta"
        assert results[0]["metadata"]["name"] == "Test Object"
        assert results[0]["metadata"]["score"] == 0.95

    def test_search_works_after_load(self, client: TestClient, temp_index_dir: Path) -> None:
        """Search works correctly after loading an index."""
        # Add distinct embeddings
        emb_a = create_normalized_embedding(DIMENSION, seed=100)
        emb_b = create_normalized_embedding(DIMENSION, seed=200)
        add_embedding(client, "obj_a", emb_a)
        add_embedding(client, "obj_b", emb_b)

        save_path = str(temp_index_dir / "searchable.index")
        client.post("/index/save", json={"path": save_path})

        # Clear and reload
        client.delete("/index")
        client.post("/index/load", json={"path": save_path})

        # Search should find correct object
        response = client.post("/search", json={"embedding": emb_a, "k": 2, "threshold": 0.0})
        results = response.json()["results"]
        assert results[0]["object_id"] == "obj_a"
        assert results[0]["score"] > 0.99

    def test_load_nonexistent_file_returns_error(
        self, client: TestClient, temp_index_dir: Path
    ) -> None:
        """Loading non-existent file returns appropriate error."""
        nonexistent_path = str(temp_index_dir / "does_not_exist.index")
        response = client.post("/index/load", json={"path": nonexistent_path})
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "load_failed"
