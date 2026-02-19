"""
Shared fixtures for integration tests.

Integration tests use the real gateway application with HTTP-level mocking
for backend services. These tests verify the full request -> processing -> response
cycle without requiring actual backend services to be running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import clear_settings_cache
from tests.factories import create_mock_embedding, create_test_image_base64

if TYPE_CHECKING:
    from collections.abc import Iterator


# Backend service URLs (must match config.yaml)
EMBEDDINGS_URL = "http://localhost:8001"
SEARCH_URL = "http://localhost:8002"
GEOMETRIC_URL = "http://localhost:8003"
STORAGE_URL = "http://localhost:8004"


@pytest.fixture(scope="module")
def integration_client() -> Iterator[TestClient]:
    """
    Create test client with the real gateway application.

    Uses context manager to trigger lifespan events (client initialization).
    This client is shared across all tests in the module.

    Note: Backend HTTP calls are mocked using respx.
    """
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    clear_settings_cache()


@pytest.fixture
def client(integration_client: TestClient) -> TestClient:
    """Alias for integration_client for test compatibility."""
    return integration_client


@pytest.fixture
def sample_image_base64() -> str:
    """Create a base64-encoded sample JPEG image."""
    return create_test_image_base64(100, 100, "red", "JPEG")


@pytest.fixture
def mock_backend_healthy() -> Iterator[respx.MockRouter]:
    """
    Mock all backend services as healthy.

    Sets up respx mocks for health endpoints of all backend services.
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings health
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        # Mock search health
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        # Mock geometric health
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        # Mock storage health
        router.get(f"{STORAGE_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        yield router


@pytest.fixture
def mock_identify_pipeline() -> Iterator[respx.MockRouter]:
    """
    Mock the full identify pipeline backend calls.

    Sets up respx mocks for:
    - Embeddings service /embed endpoint
    - Search service /search endpoint
    - All health endpoints
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings health and embed
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.post(f"{EMBEDDINGS_URL}/embed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "embedding": create_mock_embedding(768),
                    "dimension": 768,
                    "processing_time_ms": 50.0,
                    "image_id": None,
                },
            )
        )

        # Mock search health and search
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.post(f"{SEARCH_URL}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "object_id": "artwork_001",
                            "score": 0.92,
                            "rank": 1,
                            "metadata": {"name": "Mona Lisa", "artist": "Leonardo da Vinci"},
                        }
                    ],
                    "query_id": "test_query",
                    "k": 5,
                    "search_time_ms": 10.0,
                },
            )
        )

        # Mock geometric health
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        # Mock storage health and image fetches
        router.get(f"{STORAGE_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(url__startswith=f"{STORAGE_URL}/objects/", name="storage_objects").mock(
            return_value=httpx.Response(
                200,
                content=b"fake-jpeg-bytes",
                headers={"content-type": "application/octet-stream"},
            )
        )

        yield router


@pytest.fixture
def mock_storage_unavailable() -> Iterator[respx.MockRouter]:
    """
    Mock storage service as unavailable.
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings/search/geometric as healthy
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        # Mock storage as unavailable
        router.get(f"{STORAGE_URL}/health").mock(
            return_value=httpx.Response(503, json={"error": "unavailable"})
        )
        router.get(url__startswith=f"{STORAGE_URL}/objects/").mock(
            side_effect=httpx.ConnectError("storage unavailable")
        )

        yield router


@pytest.fixture
def mock_storage_missing_references() -> Iterator[respx.MockRouter]:
    """
    Mock storage service with missing reference images.
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings/search/geometric as healthy
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        # Mock storage as healthy but references missing
        router.get(f"{STORAGE_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(url__startswith=f"{STORAGE_URL}/objects/").mock(
            return_value=httpx.Response(404, json={"error": "not_found"})
        )

        yield router


@pytest.fixture
def mock_identify_no_match() -> Iterator[respx.MockRouter]:
    """
    Mock identify pipeline with no search results.
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.post(f"{EMBEDDINGS_URL}/embed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "embedding": create_mock_embedding(768),
                    "dimension": 768,
                    "processing_time_ms": 50.0,
                    "image_id": None,
                },
            )
        )

        # Mock search with empty results
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.post(f"{SEARCH_URL}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [],
                    "query_id": "test_query",
                    "k": 5,
                    "search_time_ms": 10.0,
                },
            )
        )

        # Mock geometric health
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        yield router


@pytest.fixture
def mock_embeddings_unavailable() -> Iterator[respx.MockRouter]:
    """
    Mock embeddings service as unavailable.
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings as unavailable
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(503, json={"error": "unavailable"})
        )

        # Mock other services as healthy
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        yield router
