"""
Shared fixtures for integration tests.

TODO: Add fixtures for:
    - Service URLs
    - Test images
    - HTTP client
"""

import pytest


@pytest.fixture
def gateway_url() -> str:
    """Gateway URL for integration tests."""
    return "http://localhost:8000"


@pytest.fixture
def embeddings_url() -> str:
    """Embeddings service URL for integration tests."""
    return "http://localhost:8001"


@pytest.fixture
def search_url() -> str:
    """Search service URL for integration tests."""
    return "http://localhost:8002"


@pytest.fixture
def geometric_url() -> str:
    """Geometric service URL for integration tests."""
    return "http://localhost:8003"
