"""
Shared fixtures for integration tests.

Integration tests use the real application with an actual FAISS index.
These tests verify end-to-end behavior without mocks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from search_service.app import create_app
from search_service.config import clear_settings_cache
from search_service.core.state import reset_app_state

# Re-export factory functions for backward compatibility
from tests.factories import create_normalized_embedding, create_similar_embedding

# Make factory functions available to other test modules
__all__ = ["create_normalized_embedding", "create_similar_embedding"]

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def integration_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events (index creation).
    This client is shared across all tests in the module to avoid
    recreating the app for each test.
    """
    clear_settings_cache()
    reset_app_state()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Clean up after all integration tests
    clear_settings_cache()
    reset_app_state()


@pytest.fixture
def client(integration_client: TestClient) -> Iterator[TestClient]:
    """
    Provide a clean index for each test.

    Clears the index before each test to ensure isolation.
    """
    # Clear index before each test
    integration_client.delete("/index")
    yield integration_client


@pytest.fixture
def temp_index_dir() -> Iterator[Path]:
    """
    Create a temporary directory for save/load tests.

    Yields the path to the temporary directory, which is
    automatically cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
