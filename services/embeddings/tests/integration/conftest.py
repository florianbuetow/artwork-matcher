"""
Shared fixtures for integration tests.

Integration tests use the real application with the actual model loaded.
These tests are slower but provide higher confidence in the full system.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from embeddings_service.app import create_app
from embeddings_service.config import clear_settings_cache
from embeddings_service.core.state import reset_app_state

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def integration_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events (model loading).
    This client is shared across all tests in the module to avoid
    loading the model multiple times.

    Note: We clear caches before setup and reset state after teardown,
    but NOT between individual tests (which would destroy the model).
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


@pytest.fixture
def sample_image_base64() -> str:
    """Create a minimal valid JPEG image as base64."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def sample_png_base64() -> str:
    """Create a minimal valid PNG image as base64."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def sample_webp_base64() -> str:
    """Create a minimal valid WebP image as base64."""
    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def invalid_base64() -> str:
    """Invalid base64 string for testing."""
    return "not-valid-base64!!!"


@pytest.fixture
def not_an_image_base64() -> str:
    """Valid base64 but not an image."""
    return base64.b64encode(b"not an image file").decode("ascii")


@pytest.fixture
def bmp_image_base64() -> str:
    """Create a BMP image (unsupported format) as base64."""
    img = Image.new("RGB", (10, 10), color="yellow")
    buffer = io.BytesIO()
    img.save(buffer, format="BMP")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
