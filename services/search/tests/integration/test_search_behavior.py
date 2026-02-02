"""
Integration tests for search behavior.

These tests verify that:
1. Search returns indexed items correctly (determinism)
2. Different embeddings are distinguished (discrimination)
3. Results are ranked by similarity (ranking)
4. Index lifecycle operations work (clear, re-add)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.integration.conftest import create_normalized_embedding, create_similar_embedding

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


def search(
    client: TestClient,
    embedding: list[float],
    k: int = 5,
    threshold: float = 0.0,
) -> dict:
    """Search the index."""
    response = client.post(
        "/search",
        json={
            "embedding": embedding,
            "k": k,
            "threshold": threshold,
        },
    )
    assert response.status_code == 200, f"Failed to search: {response.text}"
    return response.json()


# =============================================================================
# Search Determinism Tests
# =============================================================================


@pytest.mark.integration
class TestSearchDeterminism:
    """Tests verifying that search results are deterministic."""

    def test_same_query_returns_same_results(self, client: TestClient) -> None:
        """Same query always returns the same results."""
        # Add some items
        emb1 = create_normalized_embedding(DIMENSION, seed=100)
        emb2 = create_normalized_embedding(DIMENSION, seed=200)
        add_embedding(client, "obj_1", emb1)
        add_embedding(client, "obj_2", emb2)

        query = create_normalized_embedding(DIMENSION, seed=300)

        # Search multiple times
        results1 = search(client, query, k=2)
        results2 = search(client, query, k=2)
        results3 = search(client, query, k=2)

        # Results should be identical
        assert results1["results"] == results2["results"]
        assert results2["results"] == results3["results"]

    def test_search_for_indexed_embedding_returns_itself_first(self, client: TestClient) -> None:
        """Searching for an indexed embedding returns itself as best match."""
        # Add an embedding
        emb = create_normalized_embedding(DIMENSION, seed=42)
        add_embedding(client, "target", emb, {"name": "Target Object"})

        # Add some other embeddings
        for i in range(5):
            other_emb = create_normalized_embedding(DIMENSION, seed=1000 + i)
            add_embedding(client, f"other_{i}", other_emb)

        # Search with the target embedding
        results = search(client, emb, k=10, threshold=0.0)

        # Target should be the first result with score ~1.0
        assert results["count"] >= 1
        assert results["results"][0]["object_id"] == "target"
        assert results["results"][0]["score"] > 0.99  # Inner product of normalized vec with itself


# =============================================================================
# Search Discrimination Tests
# =============================================================================


@pytest.mark.integration
class TestSearchDiscrimination:
    """Tests verifying that different embeddings produce different results."""

    def test_different_embeddings_return_different_best_matches(self, client: TestClient) -> None:
        """Searching for different embeddings returns different best matches."""
        # Add two distinct embeddings
        emb_a = create_normalized_embedding(DIMENSION, seed=1)
        emb_b = create_normalized_embedding(DIMENSION, seed=2)
        add_embedding(client, "obj_a", emb_a)
        add_embedding(client, "obj_b", emb_b)

        # Search for each embedding
        results_a = search(client, emb_a, k=2, threshold=0.0)
        results_b = search(client, emb_b, k=2, threshold=0.0)

        # Each should return itself as best match
        assert results_a["results"][0]["object_id"] == "obj_a"
        assert results_b["results"][0]["object_id"] == "obj_b"

    def test_similar_embedding_finds_original(self, client: TestClient) -> None:
        """A similar embedding finds the original as best match."""
        # Add original embedding
        original = create_normalized_embedding(DIMENSION, seed=42)
        add_embedding(client, "original", original)

        # Add some random embeddings
        for i in range(5):
            other = create_normalized_embedding(DIMENSION, seed=1000 + i)
            add_embedding(client, f"other_{i}", other)

        # Create a similar embedding (very small perturbation for high similarity)
        similar = create_similar_embedding(original, noise_scale=0.01, seed=99)

        # Search with the similar embedding
        results = search(client, similar, k=3, threshold=0.0)

        # Original should be the best match with high similarity
        assert results["results"][0]["object_id"] == "original"
        assert results["results"][0]["score"] > 0.95  # High similarity with small noise


# =============================================================================
# Search Ranking Tests
# =============================================================================


@pytest.mark.integration
class TestSearchRanking:
    """Tests verifying that results are correctly ranked by similarity."""

    def test_results_ranked_by_descending_score(self, client: TestClient) -> None:
        """Results are returned in descending order of score."""
        # Add several embeddings
        for i in range(10):
            emb = create_normalized_embedding(DIMENSION, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        query = create_normalized_embedding(DIMENSION, seed=100)
        results = search(client, query, k=10, threshold=0.0)

        # Verify descending order
        scores = [r["score"] for r in results["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Results not in descending order: {scores[i]} < {scores[i + 1]}"
            )

    def test_k_limits_results(self, client: TestClient) -> None:
        """The k parameter limits the number of results."""
        # Create a base query and add variations of it
        # This ensures all items have positive similarity scores
        query = create_normalized_embedding(DIMENSION, seed=100)

        # Add 10 embeddings that are similar to the query
        for i in range(10):
            emb = create_similar_embedding(query, noise_scale=0.1 + i * 0.05, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        # Request different k values
        results_3 = search(client, query, k=3, threshold=0.0)
        results_5 = search(client, query, k=5, threshold=0.0)
        results_20 = search(client, query, k=20, threshold=0.0)

        assert results_3["count"] == 3
        assert results_5["count"] == 5
        assert results_20["count"] == 10  # Only 10 items in index

    def test_threshold_filters_results(self, client: TestClient) -> None:
        """Threshold filters out low-scoring results."""
        # Add an embedding and search for it (should have score ~1.0)
        emb = create_normalized_embedding(DIMENSION, seed=42)
        add_embedding(client, "high_match", emb)

        # Add random embeddings (will have lower scores)
        for i in range(5):
            other = create_normalized_embedding(DIMENSION, seed=1000 + i)
            add_embedding(client, f"other_{i}", other)

        # High threshold should only return exact/near matches
        results_high = search(client, emb, k=10, threshold=0.99)
        results_low = search(client, emb, k=10, threshold=0.0)

        assert results_high["count"] <= results_low["count"]
        # High threshold should at least return the exact match
        assert results_high["count"] >= 1
        assert results_high["results"][0]["object_id"] == "high_match"


# =============================================================================
# Index Lifecycle Tests
# =============================================================================


@pytest.mark.integration
class TestIndexLifecycle:
    """Tests verifying index lifecycle operations."""

    def test_clear_removes_all_items(self, client: TestClient) -> None:
        """Clearing the index removes all items."""
        # Create a base query and add variations that will match it
        query = create_normalized_embedding(DIMENSION, seed=100)

        # Add 5 embeddings similar to query so they have positive scores
        for i in range(5):
            emb = create_similar_embedding(query, noise_scale=0.1 + i * 0.05, seed=i)
            add_embedding(client, f"obj_{i}", emb)

        # Verify items exist (all should have positive similarity to query)
        results = search(client, query, k=10, threshold=0.0)
        assert results["count"] == 5

        # Clear the index
        response = client.delete("/index")
        assert response.status_code == 200
        clear_result = response.json()
        assert clear_result["previous_count"] == 5
        assert clear_result["current_count"] == 0

    def test_search_empty_index_returns_error(self, client: TestClient) -> None:
        """Searching an empty index returns appropriate error."""
        # Index is already cleared by fixture
        query = create_normalized_embedding(DIMENSION, seed=100)
        response = client.post(
            "/search",
            json={"embedding": query, "k": 5},
        )

        assert response.status_code == 422
        assert response.json()["error"] == "index_empty"

    def test_can_add_after_clear(self, client: TestClient) -> None:
        """Can add items after clearing the index."""
        # Add, clear, then add again
        emb1 = create_normalized_embedding(DIMENSION, seed=1)
        add_embedding(client, "before_clear", emb1)

        client.delete("/index")

        emb2 = create_normalized_embedding(DIMENSION, seed=2)
        result = add_embedding(client, "after_clear", emb2)

        assert result["index_position"] == 0  # First item in fresh index
        assert result["index_count"] == 1

        # Verify search works
        results = search(client, emb2, k=1, threshold=0.0)
        assert results["count"] == 1
        assert results["results"][0]["object_id"] == "after_clear"
