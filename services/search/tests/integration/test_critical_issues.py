"""
Tests exposing critical issues identified in PR review.

These tests verify:
1. Path traversal vulnerability in save/load endpoints
2. max_k configuration not enforced
3. Silent startup failure masking index load errors
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.factories import create_normalized_embedding

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# =============================================================================
# Test Helpers
# =============================================================================


def add_embedding(client: TestClient, object_id: str, embedding: list[float]) -> dict:
    """Helper to add an embedding to the index."""
    response = client.post(
        "/add",
        json={"object_id": object_id, "embedding": embedding},
    )
    return response.json()


def search(client: TestClient, embedding: list[float], k: int | None = None) -> dict:
    """Helper to search the index."""
    payload: dict = {"embedding": embedding}
    if k is not None:
        payload["k"] = k
    response = client.post("/search", json=payload)
    return {"status_code": response.status_code, **response.json()}


# =============================================================================
# Issue 1: Path Traversal Vulnerability
# =============================================================================


@pytest.mark.integration
class TestPathTraversalVulnerability:
    """Tests that path traversal attacks are rejected in save/load endpoints."""

    def test_save_rejects_path_traversal_with_parent_directories(
        self, client: TestClient, temp_index_dir: Path
    ) -> None:
        """
        Saving to a path with '..' should be rejected.

        Currently the endpoint accepts any path, allowing writes outside
        the allowed index directory. This is a security vulnerability.
        """
        # Add an embedding so we have something to save
        embedding = create_normalized_embedding(768, seed=42)
        add_embedding(client, "test_object", embedding)

        # Try to save to a path that traverses outside allowed directory
        malicious_path = str(temp_index_dir / ".." / ".." / "etc" / "malicious.index")

        response = client.post("/index/save", json={"path": malicious_path})

        # Should reject with 400, not succeed
        assert response.status_code == 400, (
            f"Path traversal should be rejected. "
            f"Got status {response.status_code}, response: {response.json()}"
        )
        data = response.json()
        assert data["error"] == "path_not_allowed"

    def test_save_rejects_absolute_path_outside_allowed_dir(self, client: TestClient) -> None:
        """
        Saving to an absolute path outside allowed directory should be rejected.

        This tests that arbitrary absolute paths cannot be used to write
        files anywhere on the filesystem.
        """
        embedding = create_normalized_embedding(768, seed=42)
        add_embedding(client, "test_object", embedding)

        # Try to save to /tmp (outside normal index directory)
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_path = str(Path(tmpdir) / "stolen.index")

            response = client.post("/index/save", json={"path": malicious_path})

            # Should reject with 400
            assert response.status_code == 400, (
                f"Absolute path outside allowed dir should be rejected. "
                f"Got status {response.status_code}"
            )

    def test_load_rejects_path_traversal_with_parent_directories(
        self, client: TestClient, temp_index_dir: Path
    ) -> None:
        """
        Loading from a path with '..' should be rejected.

        This could allow reading arbitrary files from the filesystem.
        """
        malicious_path = str(temp_index_dir / ".." / ".." / "etc" / "passwd.index")

        response = client.post("/index/load", json={"path": malicious_path})

        # Should reject with 400, not 500 (file not found)
        assert response.status_code == 400, (
            f"Path traversal in load should be rejected with 400. Got status {response.status_code}"
        )
        data = response.json()
        assert data["error"] == "path_not_allowed"

    def test_load_rejects_absolute_path_outside_allowed_dir(self, client: TestClient) -> None:
        """
        Loading from an absolute path outside allowed directory should be rejected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_path = str(Path(tmpdir) / "fake.index")

            response = client.post("/index/load", json={"path": malicious_path})

            # Should reject with 400
            assert response.status_code == 400, (
                f"Absolute path outside allowed dir should be rejected. "
                f"Got status {response.status_code}"
            )


# =============================================================================
# Issue 2: max_k Configuration Not Enforced
# =============================================================================


@pytest.mark.integration
class TestMaxKEnforcement:
    """Tests that max_k configuration is enforced in search endpoint."""

    def test_search_rejects_k_exceeding_config_max_k(self, client: TestClient) -> None:
        """
        Searching with k > config.search.max_k should be rejected.

        The config defines max_k: 100, but currently the schema has a
        hardcoded le=100 and the endpoint doesn't enforce the config value.
        This test verifies the config value is actually used.
        """
        # Add some embeddings to make index non-empty
        for i in range(3):
            embedding = create_normalized_embedding(768, seed=i)
            add_embedding(client, f"object_{i}", embedding)

        # Try to search with k=150 (exceeds max_k=100)
        query = create_normalized_embedding(768, seed=999)
        result = search(client, query, k=150)

        # Should reject with 400, not succeed
        # Note: Currently schema validates le=100, so this test will fail
        # at schema validation, but it should be rejected based on config.max_k
        assert result["status_code"] == 400, (
            f"k exceeding max_k should be rejected. Got status {result['status_code']}"
        )
        assert result["error"] == "k_exceeds_maximum"

    def test_search_accepts_k_at_config_max_k(self, client: TestClient) -> None:
        """
        Searching with k exactly at config.search.max_k should succeed.
        """
        # Add embeddings
        for i in range(3):
            embedding = create_normalized_embedding(768, seed=i)
            add_embedding(client, f"object_{i}", embedding)

        # Search with k=100 (exactly max_k)
        query = create_normalized_embedding(768, seed=999)
        result = search(client, query, k=100)

        # Should succeed
        assert result["status_code"] == 200, (
            f"k at max_k should succeed. Got status {result['status_code']}"
        )

    def test_search_error_includes_max_k_in_details(self, client: TestClient) -> None:
        """
        When k exceeds max_k, error details should include the configured max_k.

        This helps users understand the limit.
        """
        for i in range(3):
            embedding = create_normalized_embedding(768, seed=i)
            add_embedding(client, f"object_{i}", embedding)

        query = create_normalized_embedding(768, seed=999)
        result = search(client, query, k=150)

        assert result["status_code"] == 400
        assert "details" in result
        assert "max_k" in result["details"]
        assert result["details"]["max_k"] == 100  # From config


# =============================================================================
# Issue 3: Silent Startup Failure
# =============================================================================


@pytest.mark.integration
class TestStartupFailureVisibility:
    """Tests that index load failures at startup are visible, not silent."""

    def test_health_indicates_failed_index_load(self) -> None:
        """
        When index auto-load fails, health endpoint should indicate degraded state.

        Currently, if the index file exists but is corrupted, the service
        silently starts with an empty index. The health endpoint shows
        "healthy" even though the expected data failed to load.
        """
        # This test requires a custom app setup with a corrupted index file
        # We'll create a corrupted index file, set auto_load=true, and verify
        # that the health endpoint indicates the problem

        import os
        import tempfile

        from fastapi.testclient import TestClient

        from search_service.app import create_app
        from search_service.config import clear_settings_cache
        from search_service.core.state import reset_app_state

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted index file
            index_path = Path(tmpdir) / "faiss.index"
            metadata_path = Path(tmpdir) / "metadata.json"

            # Write invalid data to index file (not a valid FAISS index)
            index_path.write_bytes(b"corrupted data that is not a valid FAISS index")
            # Write valid metadata structure
            metadata_path.write_text(json.dumps({"items": []}))

            # Create custom config pointing to the corrupted files
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(f"""
service:
  name: "search"
  version: "0.1.0"

faiss:
  embedding_dimension: 768
  index_type: "flat"
  metric: "inner_product"

index:
  path: "{index_path}"
  metadata_path: "{metadata_path}"
  auto_load: true

search:
  default_k: 5
  max_k: 100
  default_threshold: 0.0

server:
  host: "0.0.0.0"
  port: 8002

logging:
  level: "INFO"
  format: "json"
""")

            # Set CONFIG_PATH to use our custom config
            old_config_path = os.environ.get("CONFIG_PATH")
            os.environ["CONFIG_PATH"] = str(config_path)

            try:
                clear_settings_cache()
                reset_app_state()
                app = create_app()

                with TestClient(app) as test_client:
                    # Check health endpoint
                    response = test_client.get("/health")
                    data = response.json()

                    # Health should indicate degraded state, not healthy
                    # Currently it shows "healthy" even with failed load
                    assert data["status"] in ("degraded", "unhealthy"), (
                        f"Health should indicate degraded/unhealthy when index "
                        f"load failed. Got status: {data['status']}"
                    )

            finally:
                # Restore original config path
                if old_config_path is not None:
                    os.environ["CONFIG_PATH"] = old_config_path
                else:
                    os.environ.pop("CONFIG_PATH", None)
                clear_settings_cache()
                reset_app_state()

    def test_info_shows_load_failure_reason(self) -> None:
        """
        When index auto-load fails, info endpoint should expose the error.

        This allows operators to understand why the index is empty when
        they expected it to be populated.
        """
        import os
        import tempfile

        from fastapi.testclient import TestClient

        from search_service.app import create_app
        from search_service.config import clear_settings_cache
        from search_service.core.state import reset_app_state

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted index file
            index_path = Path(tmpdir) / "faiss.index"
            metadata_path = Path(tmpdir) / "metadata.json"

            index_path.write_bytes(b"corrupted")
            metadata_path.write_text(json.dumps({"items": []}))

            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(f"""
service:
  name: "search"
  version: "0.1.0"

faiss:
  embedding_dimension: 768
  index_type: "flat"
  metric: "inner_product"

index:
  path: "{index_path}"
  metadata_path: "{metadata_path}"
  auto_load: true

search:
  default_k: 5
  max_k: 100
  default_threshold: 0.0

server:
  host: "0.0.0.0"
  port: 8002

logging:
  level: "INFO"
  format: "json"
""")

            old_config_path = os.environ.get("CONFIG_PATH")
            os.environ["CONFIG_PATH"] = str(config_path)

            try:
                clear_settings_cache()
                reset_app_state()
                app = create_app()

                with TestClient(app) as test_client:
                    response = test_client.get("/info")
                    data = response.json()

                    # Info should include load failure information
                    # Currently it doesn't expose this
                    assert "load_error" in data or (
                        "index" in data and "load_error" in data["index"]
                    ), f"Info should expose index load failure reason. Got: {data}"

            finally:
                if old_config_path is not None:
                    os.environ["CONFIG_PATH"] = old_config_path
                else:
                    os.environ.pop("CONFIG_PATH", None)
                clear_settings_cache()
                reset_app_state()
