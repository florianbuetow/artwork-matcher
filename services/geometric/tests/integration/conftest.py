"""
Fixtures for integration tests.

These tests use the real application with actual OpenCV processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from geometric_service.app import create_app
from geometric_service.config import clear_settings_cache
from geometric_service.core.state import reset_app_state

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def integration_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events (state initialization).
    This client is shared across all tests in the module to avoid
    repeated initialization.
    """
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Clean up after all tests in this module
    clear_settings_cache()
    reset_app_state()


@pytest.fixture
def client(integration_client: TestClient) -> TestClient:
    """
    Alias for integration_client for backward compatibility.

    Individual tests can use this fixture name.
    """
    return integration_client
