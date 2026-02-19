"""
Shared fixtures for unit tests.

All external dependencies (backend clients, settings, app state) are mocked
to ensure tests run in isolation without I/O or network access.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.factories import create_mock_embedding, create_test_image_base64

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Generator[None, None, None]:
    """Reset settings cache before and after each test."""
    from gateway.config import clear_settings_cache  # noqa: PLC0415

    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def test_config_file() -> Generator[Path, None, None]:
    """Create a temporary config file for testing."""
    config_content = """
service:
  name: "gateway"
  version: "0.1.0"

backends:
  embeddings_url: "http://localhost:8001"
  search_url: "http://localhost:8002"
  geometric_url: "http://localhost:8003"
  storage_url: "http://localhost:8004"
  timeout_seconds: 30.0
  retry:
    max_attempts: 3
    initial_backoff_seconds: 0.1
    max_backoff_seconds: 1.0
    jitter_seconds: 0.05
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout_seconds: 15.0

pipeline:
  search_k: 5
  similarity_threshold: 0.7
  geometric_verification: true
  confidence_threshold: 0.6

scoring:
  geometric_score_threshold: 0.5
  geometric_high_similarity_weight: 0.6
  geometric_high_score_weight: 0.4
  geometric_low_similarity_weight: 0.3
  geometric_low_score_weight: 0.2
  geometric_missing_penalty: 0.7
  embedding_only_penalty: 0.85

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
  cors_origins:
    - "*"

data:
  labels_path: "/tmp/test_labels.csv"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()
        config_path = Path(f.name)

    # Set environment variable to use this config
    old_config_path = os.environ.get("CONFIG_PATH")
    os.environ["CONFIG_PATH"] = str(config_path)

    yield config_path

    # Cleanup
    if old_config_path:
        os.environ["CONFIG_PATH"] = old_config_path
    else:
        os.environ.pop("CONFIG_PATH", None)

    config_path.unlink(missing_ok=True)


@pytest.fixture
def mock_embeddings_client() -> AsyncMock:
    """Create a mock embeddings client."""
    client = AsyncMock()
    client.health_check.return_value = "healthy"
    client.embed.return_value = create_mock_embedding(768)
    client.get_info.return_value = {
        "service": "embeddings",
        "version": "0.1.0",
        "model": {
            "name": "facebook/dinov2-base",
            "embedding_dimension": 768,
            "device": "cpu",
        },
    }
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_search_client() -> AsyncMock:
    """Create a mock search client."""
    client = AsyncMock()
    client.health_check.return_value = "healthy"

    # Create mock search results
    mock_result = MagicMock()
    mock_result.object_id = "object_001"
    mock_result.score = 0.92
    mock_result.rank = 1
    mock_result.metadata = {"name": "Test Artwork", "artist": "Test Artist"}

    client.search.return_value = [mock_result]
    client.get_info.return_value = {
        "service": "search",
        "version": "0.1.0",
        "index": {
            "type": "flat",
            "metric": "inner_product",
            "embedding_dimension": 768,
            "count": 20,
            "is_loaded": True,
        },
    }
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_geometric_client() -> AsyncMock:
    """Create a mock geometric client."""
    client = AsyncMock()
    client.health_check.return_value = "healthy"
    client.get_info.return_value = {
        "service": "geometric",
        "version": "0.1.0",
        "algorithm": {
            "feature_detector": "ORB",
            "max_features": 1000,
        },
    }
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_storage_client() -> AsyncMock:
    """Create a mock storage client."""
    client = AsyncMock()
    client.health_check.return_value = "healthy"
    client.get_image_bytes.return_value = b"fake jpeg data"
    client.get_image_base64.return_value = "ZmFrZSBqcGVnIGRhdGE="
    client.get_info.return_value = {
        "service": "storage",
        "version": "0.1.0",
        "objects": {"count": 28},
    }
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_app_state(
    mock_embeddings_client: AsyncMock,
    mock_search_client: AsyncMock,
    mock_geometric_client: AsyncMock,
    mock_storage_client: AsyncMock,
) -> MagicMock:
    """Create mock app state with mock clients."""
    mock_state = MagicMock()
    mock_state.embeddings_client = mock_embeddings_client
    mock_state.search_client = mock_search_client
    mock_state.geometric_client = mock_geometric_client
    mock_state.storage_client = mock_storage_client
    mock_state.uptime_seconds = 123.45
    mock_state.uptime_formatted = "2m 3s"
    return mock_state


@pytest.fixture
def test_client(
    test_config_file: Path,  # noqa: ARG001 - fixture needed for side effects
    mock_app_state: MagicMock,
) -> Generator[TestClient, None, None]:
    """Create a test client with mocked dependencies."""
    # Local imports required to avoid import order issues with config
    from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415

    from gateway.config import get_settings  # noqa: PLC0415
    from gateway.core.exceptions import register_exception_handlers  # noqa: PLC0415
    from gateway.routers import health, identify, info, objects  # noqa: PLC0415

    # Create a minimal lifespan that does nothing
    @asynccontextmanager
    async def mock_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield

    settings = get_settings()

    # Create app without the real lifespan
    app = FastAPI(
        title=f"{settings.service.name} Service",
        description="API gateway for artwork identification",
        version=settings.service.version,
        lifespan=mock_lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routers
    app.include_router(health.router, tags=["Operations"])
    app.include_router(info.router, tags=["Operations"])
    app.include_router(identify.router, tags=["Identification"])
    app.include_router(objects.router, tags=["Objects"])

    # Patch get_app_state to return our mock
    with (
        patch("gateway.core.state.get_app_state", return_value=mock_app_state),
        patch("gateway.routers.health.get_app_state", return_value=mock_app_state),
        patch("gateway.routers.info.get_app_state", return_value=mock_app_state),
        patch("gateway.routers.identify.get_app_state", return_value=mock_app_state),
        patch("gateway.routers.objects.get_app_state", return_value=mock_app_state),
        TestClient(app) as client,
    ):
        yield client


@pytest.fixture
def sample_image_base64() -> str:
    """Create a base64-encoded sample JPEG image."""
    return create_test_image_base64(100, 100, "red", "JPEG")


@pytest.fixture
def sample_png_base64() -> str:
    """Create a base64-encoded sample PNG image."""
    return create_test_image_base64(100, 100, "blue", "PNG")
