"""
Shared fixtures for integration tests.

Integration tests use the real application with an actual FAISS index.
These tests verify end-to-end behavior without mocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi.testclient import TestClient

from search_service.app import create_app
from search_service.config import clear_settings_cache
from search_service.core.state import reset_app_state

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


def create_normalized_embedding(dimension: int, seed: int | None = None) -> list[float]:
    """
    Create a normalized random embedding.

    For inner product search, normalized vectors ensure similarity
    scores are in the range [-1, 1] (cosine similarity).

    Args:
        dimension: Vector dimension
        seed: Random seed for reproducibility

    Returns:
        L2-normalized embedding as a list of floats
    """
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randn(dimension).astype(np.float32)
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()


def create_similar_embedding(
    base: list[float], noise_scale: float = 0.1, seed: int | None = None
) -> list[float]:
    """
    Create an embedding similar to the base with added noise.

    Args:
        base: Base embedding to modify
        noise_scale: Scale of noise to add (smaller = more similar)
        seed: Random seed for reproducibility

    Returns:
        L2-normalized embedding similar to base
    """
    if seed is not None:
        np.random.seed(seed)
    arr = np.array(base, dtype=np.float32)
    noise = np.random.randn(len(base)).astype(np.float32) * noise_scale
    arr = arr + noise
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()
