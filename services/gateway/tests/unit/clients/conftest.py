"""Fixtures for client unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from gateway.clients.base import BackendClient
    from gateway.clients.embeddings import EmbeddingsClient
    from gateway.clients.geometric import GeometricClient
    from gateway.clients.search import SearchClient


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    """Create a mock httpx response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {}
    return response


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient for testing client methods."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
async def embeddings_client(mock_httpx_client: AsyncMock) -> AsyncGenerator[EmbeddingsClient, None]:
    """Create an EmbeddingsClient with a mocked httpx client."""
    from gateway.clients.embeddings import EmbeddingsClient  # noqa: PLC0415

    client = EmbeddingsClient(
        base_url="http://localhost:8001",
        timeout=30.0,
        service_name="embeddings",
    )
    # Replace the internal httpx client with our mock
    client.client = mock_httpx_client
    yield client


@pytest.fixture
async def search_client(mock_httpx_client: AsyncMock) -> AsyncGenerator[SearchClient, None]:
    """Create a SearchClient with a mocked httpx client."""
    from gateway.clients.search import SearchClient  # noqa: PLC0415

    client = SearchClient(
        base_url="http://localhost:8002",
        timeout=30.0,
        service_name="search",
    )
    # Replace the internal httpx client with our mock
    client.client = mock_httpx_client
    yield client


@pytest.fixture
async def geometric_client(mock_httpx_client: AsyncMock) -> AsyncGenerator[GeometricClient, None]:
    """Create a GeometricClient with a mocked httpx client."""
    from gateway.clients.geometric import GeometricClient  # noqa: PLC0415

    client = GeometricClient(
        base_url="http://localhost:8003",
        timeout=30.0,
        service_name="geometric",
    )
    # Replace the internal httpx client with our mock
    client.client = mock_httpx_client
    yield client


@pytest.fixture
async def backend_client(mock_httpx_client: AsyncMock) -> AsyncGenerator[BackendClient, None]:
    """Create a generic BackendClient with a mocked httpx client."""
    from gateway.clients.base import BackendClient  # noqa: PLC0415

    client = BackendClient(
        base_url="http://localhost:8000",
        timeout=30.0,
        service_name="test_backend",
    )
    # Replace the internal httpx client with our mock
    client.client = mock_httpx_client
    yield client
