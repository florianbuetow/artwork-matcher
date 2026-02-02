"""
Integration tests for error responses.

These tests verify that the API returns correct error codes,
messages, and details for various error conditions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.factories import (
    create_normalized_embedding,
    create_wrong_dimension_embedding,
)

if TYPE_CHECKING:
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
# Dimension Mismatch Error Tests
# =============================================================================


@pytest.mark.integration
class TestDimensionMismatchErrors:
    """Tests for dimension mismatch error handling."""

    def test_add_wrong_dimension_returns_400(self, client: TestClient) -> None:
        """Adding embedding with wrong dimension returns 400."""
        wrong_dim_embedding = create_wrong_dimension_embedding(DIMENSION)
        response = client.post(
            "/add",
            json={
                "object_id": "test_obj",
                "embedding": wrong_dim_embedding,
            },
        )
        assert response.status_code == 400

    def test_add_wrong_dimension_error_code(self, client: TestClient) -> None:
        """Dimension mismatch error has correct error code."""
        wrong_dim_embedding = create_wrong_dimension_embedding(DIMENSION)
        response = client.post(
            "/add",
            json={
                "object_id": "test_obj",
                "embedding": wrong_dim_embedding,
            },
        )
        data = response.json()
        assert data["error"] == "dimension_mismatch"

    def test_add_wrong_dimension_includes_details(self, client: TestClient) -> None:
        """Dimension mismatch error includes expected and received dimensions."""
        wrong_dim_embedding = create_wrong_dimension_embedding(DIMENSION)
        response = client.post(
            "/add",
            json={
                "object_id": "test_obj",
                "embedding": wrong_dim_embedding,
            },
        )
        data = response.json()
        assert "details" in data
        assert data["details"]["expected"] == DIMENSION
        assert data["details"]["received"] == DIMENSION - 1

    def test_search_wrong_dimension_returns_400(self, client: TestClient) -> None:
        """Searching with wrong dimension returns 400."""
        # First add a valid embedding
        valid_emb = create_normalized_embedding(DIMENSION, seed=1)
        add_embedding(client, "obj_1", valid_emb)

        # Search with wrong dimension
        wrong_dim_embedding = create_wrong_dimension_embedding(DIMENSION)
        response = client.post(
            "/search",
            json={
                "embedding": wrong_dim_embedding,
                "k": 5,
            },
        )
        assert response.status_code == 400
        assert response.json()["error"] == "dimension_mismatch"


# =============================================================================
# Invalid Embedding Error Tests
# =============================================================================

# NOTE: Tests for NaN/Inf embeddings cannot be performed via HTTP because
# standard JSON does not support these values. The JSON encoder will raise
# "ValueError: Out of range float values are not JSON compliant".
#
# Invalid embedding validation (NaN, Inf, zero vectors) is tested via unit
# tests that directly call the FaissIndex methods.
#
# See: tests/unit/test_faiss_index.py for invalid embedding tests


# =============================================================================
# Index Empty Error Tests
# =============================================================================


@pytest.mark.integration
class TestIndexEmptyError:
    """Tests for empty index error handling."""

    def test_search_empty_index_returns_422(self, client: TestClient) -> None:
        """Searching an empty index returns 422."""
        # Index is cleared by fixture
        query = create_normalized_embedding(DIMENSION, seed=100)
        response = client.post(
            "/search",
            json={
                "embedding": query,
                "k": 5,
            },
        )
        assert response.status_code == 422

    def test_search_empty_error_code(self, client: TestClient) -> None:
        """Empty index error has correct error code."""
        query = create_normalized_embedding(DIMENSION, seed=100)
        response = client.post(
            "/search",
            json={
                "embedding": query,
                "k": 5,
            },
        )
        data = response.json()
        assert data["error"] == "index_empty"

    def test_search_empty_error_message(self, client: TestClient) -> None:
        """Empty index error has informative message."""
        query = create_normalized_embedding(DIMENSION, seed=100)
        response = client.post(
            "/search",
            json={
                "embedding": query,
                "k": 5,
            },
        )
        data = response.json()
        assert "message" in data
        assert "empty" in data["message"].lower()

    def test_search_empty_error_details(self, client: TestClient) -> None:
        """Empty index error includes count in details."""
        query = create_normalized_embedding(DIMENSION, seed=100)
        response = client.post(
            "/search",
            json={
                "embedding": query,
                "k": 5,
            },
        )
        data = response.json()
        assert "details" in data
        assert data["details"]["count"] == 0
