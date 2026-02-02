"""
Integration tests with deterministic vectors and known expected results.

These tests use precisely constructed vectors where we know the exact
similarity scores and expected rankings in advance. This allows us to
verify that search returns exactly what we expect, not just reasonable results.

Key concepts:
- Basis vectors are orthogonal: similarity = 0
- Identical vectors: similarity = 1.0
- Opposite vectors: similarity = -1.0
- Vectors at known angles: predictable similarity
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.factories import (
    compute_inner_product,
    create_basis_vector,
    create_opposite_vector,
    create_scaled_similarity_vectors,
    create_vector_with_known_similarity,
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
# Exact Match Tests
# =============================================================================


@pytest.mark.integration
class TestExactMatchSearch:
    """Tests verifying exact match behavior with identical vectors."""

    def test_identical_vector_returns_score_one(self, client: TestClient) -> None:
        """Searching for an identical vector returns score of 1.0."""
        vec = create_basis_vector(DIMENSION, 0)
        add_embedding(client, "exact", vec)

        results = search(client, vec, k=1, threshold=0.0)

        assert results["count"] == 1
        assert results["results"][0]["object_id"] == "exact"
        # Score should be exactly 1.0 for identical normalized vectors
        assert abs(results["results"][0]["score"] - 1.0) < 0.001

    def test_multiple_exact_matches_all_score_one(self, client: TestClient) -> None:
        """Multiple identical vectors all return score 1.0."""
        vec = create_basis_vector(DIMENSION, 0)

        # Add the same vector multiple times with different IDs
        add_embedding(client, "copy_a", vec)
        add_embedding(client, "copy_b", vec)
        add_embedding(client, "copy_c", vec)

        results = search(client, vec, k=3, threshold=0.0)

        assert results["count"] == 3
        for result in results["results"]:
            assert abs(result["score"] - 1.0) < 0.001


# =============================================================================
# Orthogonal Vector Tests
# =============================================================================


@pytest.mark.integration
class TestOrthogonalVectorSearch:
    """Tests verifying behavior with orthogonal (perpendicular) vectors."""

    def test_orthogonal_vectors_have_zero_similarity(self, client: TestClient) -> None:
        """Orthogonal basis vectors have similarity score of 0."""
        vec_0 = create_basis_vector(DIMENSION, 0)
        vec_1 = create_basis_vector(DIMENSION, 1)

        # Verify they are orthogonal
        assert abs(compute_inner_product(vec_0, vec_1)) < 0.001

        add_embedding(client, "orthogonal", vec_1)

        # threshold=0.0 should include vectors with score >= 0
        results = search(client, vec_0, k=1, threshold=0.0)

        assert results["count"] == 1
        assert results["results"][0]["object_id"] == "orthogonal"
        assert abs(results["results"][0]["score"]) < 0.001  # Should be ~0

    def test_threshold_filters_orthogonal_vectors(self, client: TestClient) -> None:
        """Positive threshold filters out orthogonal vectors (score=0)."""
        vec_0 = create_basis_vector(DIMENSION, 0)
        vec_1 = create_basis_vector(DIMENSION, 1)

        add_embedding(client, "orthogonal", vec_1)

        # With threshold > 0, orthogonal vector should be filtered
        results = search(client, vec_0, k=1, threshold=0.1)

        assert results["count"] == 0


# =============================================================================
# Known Similarity Tests
# =============================================================================


@pytest.mark.integration
class TestKnownSimilaritySearch:
    """Tests with vectors constructed to have specific similarity scores."""

    def test_vector_with_known_similarity_returns_expected_score(self, client: TestClient) -> None:
        """Vector created with target similarity returns that score."""
        base = create_basis_vector(DIMENSION, 0)
        target_sim = 0.75
        similar_vec = create_vector_with_known_similarity(base, target_sim)

        # Verify the factory produced correct similarity
        actual_sim = compute_inner_product(base, similar_vec)
        assert abs(actual_sim - target_sim) < 0.001

        add_embedding(client, "similar", similar_vec)

        results = search(client, base, k=1, threshold=0.0)

        assert results["count"] == 1
        assert results["results"][0]["object_id"] == "similar"
        assert abs(results["results"][0]["score"] - target_sim) < 0.01

    def test_opposite_vector_filtered_by_zero_threshold(self, client: TestClient) -> None:
        """Opposite (negated) vector is filtered out by threshold >= 0."""
        base = create_basis_vector(DIMENSION, 0)
        opposite = create_opposite_vector(base)

        # Verify they are opposite (similarity = -1.0)
        assert abs(compute_inner_product(base, opposite) + 1.0) < 0.001

        add_embedding(client, "opposite", opposite)

        # API threshold minimum is 0.0, so negative scores are always filtered
        results = search(client, base, k=1, threshold=0.0)

        # Opposite vector (score=-1.0) should be filtered out
        assert results["count"] == 0


# =============================================================================
# Ranking Tests with Known Order
# =============================================================================


@pytest.mark.integration
class TestKnownRankingOrder:
    """Tests verifying exact ranking order with known similarities."""

    def test_three_vectors_rank_by_similarity(self, client: TestClient) -> None:
        """Three vectors with known similarities rank correctly."""
        base, vectors = create_scaled_similarity_vectors(DIMENSION, [0.9, 0.5, 0.2])

        # Add vectors in scrambled order (not by similarity)
        add_embedding(client, "medium", vectors[1][1], {"expected_rank": 2})
        add_embedding(client, "low", vectors[2][1], {"expected_rank": 3})
        add_embedding(client, "high", vectors[0][1], {"expected_rank": 1})

        results = search(client, base, k=3, threshold=0.0)

        assert results["count"] == 3
        # Verify exact ranking order
        assert results["results"][0]["object_id"] == "high"
        assert results["results"][1]["object_id"] == "medium"
        assert results["results"][2]["object_id"] == "low"

        # Verify scores match expected similarities
        assert abs(results["results"][0]["score"] - 0.9) < 0.02
        assert abs(results["results"][1]["score"] - 0.5) < 0.02
        assert abs(results["results"][2]["score"] - 0.2) < 0.02

    def test_five_vectors_exact_ranking(self, client: TestClient) -> None:
        """Five vectors rank in exact expected order."""
        similarities = [0.95, 0.80, 0.60, 0.40, 0.20]
        base, vectors = create_scaled_similarity_vectors(DIMENSION, similarities)

        # Add with identifiable names
        ids = ["first", "second", "third", "fourth", "fifth"]
        for i, (sim, vec) in enumerate(vectors):
            add_embedding(client, ids[i], vec, {"similarity": sim})

        results = search(client, base, k=5, threshold=0.0)

        assert results["count"] == 5

        # Verify exact order
        for i, expected_id in enumerate(ids):
            assert results["results"][i]["object_id"] == expected_id
            assert results["results"][i]["rank"] == i + 1

    def test_k_returns_top_k_by_similarity(self, client: TestClient) -> None:
        """Requesting k results returns exactly the top k by similarity."""
        similarities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        base, vectors = create_scaled_similarity_vectors(DIMENSION, similarities)

        # Add all vectors
        for i, (_sim, vec) in enumerate(vectors):
            add_embedding(client, f"vec_{i}", vec)

        # Request top 3
        results = search(client, base, k=3, threshold=0.0)

        assert results["count"] == 3
        assert results["results"][0]["object_id"] == "vec_0"  # sim=0.9
        assert results["results"][1]["object_id"] == "vec_1"  # sim=0.8
        assert results["results"][2]["object_id"] == "vec_2"  # sim=0.7


# =============================================================================
# Threshold Filtering Tests with Known Scores
# =============================================================================


@pytest.mark.integration
class TestThresholdFiltering:
    """Tests verifying threshold filtering with known similarity scores."""

    def test_threshold_excludes_below_cutoff(self, client: TestClient) -> None:
        """Threshold excludes vectors with similarity below cutoff."""
        similarities = [0.9, 0.7, 0.5, 0.3, 0.1]
        base, vectors = create_scaled_similarity_vectors(DIMENSION, similarities)

        for i, (sim, vec) in enumerate(vectors):
            add_embedding(client, f"vec_{i}", vec, {"similarity": sim})

        # Threshold at 0.6 should return only 0.9 and 0.7
        results = search(client, base, k=10, threshold=0.6)

        assert results["count"] == 2
        assert results["results"][0]["object_id"] == "vec_0"  # sim=0.9
        assert results["results"][1]["object_id"] == "vec_1"  # sim=0.7

    def test_threshold_exactly_at_score_includes_it(self, client: TestClient) -> None:
        """Vector with score exactly at threshold is included."""
        base = create_basis_vector(DIMENSION, 0)
        vec_at_threshold = create_vector_with_known_similarity(base, 0.5)

        add_embedding(client, "at_threshold", vec_at_threshold)

        # Threshold at exactly the similarity score
        results = search(client, base, k=1, threshold=0.5)

        # Should be included (threshold is inclusive)
        assert results["count"] == 1
        assert results["results"][0]["object_id"] == "at_threshold"

    def test_threshold_just_above_score_excludes_it(self, client: TestClient) -> None:
        """Vector with score just below threshold is excluded."""
        base = create_basis_vector(DIMENSION, 0)
        vec_below = create_vector_with_known_similarity(base, 0.49)

        add_embedding(client, "below_threshold", vec_below)

        # Threshold just above the similarity
        results = search(client, base, k=1, threshold=0.5)

        assert results["count"] == 0

    def test_high_threshold_returns_only_near_exact_matches(self, client: TestClient) -> None:
        """Very high threshold (0.99) only returns near-exact matches."""
        base = create_basis_vector(DIMENSION, 0)

        # Add exact match
        add_embedding(client, "exact", base.copy())

        # Add very similar but not exact
        almost = create_vector_with_known_similarity(base, 0.98)
        add_embedding(client, "almost", almost)

        results = search(client, base, k=10, threshold=0.99)

        # Only exact match should pass 0.99 threshold
        assert results["count"] == 1
        assert results["results"][0]["object_id"] == "exact"


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.integration
class TestDeterministicEdgeCases:
    """Edge case tests with deterministic vectors."""

    def test_search_with_all_orthogonal_vectors(self, client: TestClient) -> None:
        """Search among orthogonal vectors returns all with score ~0."""
        query = create_basis_vector(DIMENSION, 0)

        # Add vectors orthogonal to query
        for i in range(1, 6):
            orthogonal = create_basis_vector(DIMENSION, i)
            add_embedding(client, f"orth_{i}", orthogonal)

        # threshold=0.0 includes vectors with score >= 0
        results = search(client, query, k=5, threshold=0.0)

        assert results["count"] == 5
        # All should have score ~0
        for result in results["results"]:
            assert abs(result["score"]) < 0.001

    def test_negative_similarities_filtered_by_threshold(self, client: TestClient) -> None:
        """Vectors with negative similarity are filtered out by threshold >= 0."""
        base = create_basis_vector(DIMENSION, 0)

        # Create vectors with positive and negative similarities
        pos_high = create_vector_with_known_similarity(base, 0.8)
        pos_low = create_vector_with_known_similarity(base, 0.3)
        neg_low = create_vector_with_known_similarity(base, -0.3)
        neg_high = create_opposite_vector(base)  # -1.0

        add_embedding(client, "pos_high", pos_high)
        add_embedding(client, "pos_low", pos_low)
        add_embedding(client, "neg_low", neg_low)
        add_embedding(client, "neg_high", neg_high)

        # API minimum threshold is 0.0, so negative scores are filtered
        results = search(client, base, k=4, threshold=0.0)

        # Only positive similarities should be returned
        assert results["count"] == 2
        assert results["results"][0]["object_id"] == "pos_high"
        assert results["results"][1]["object_id"] == "pos_low"

    def test_many_vectors_with_same_similarity(self, client: TestClient) -> None:
        """Multiple vectors with identical similarity are all returned."""
        base = create_basis_vector(DIMENSION, 0)

        # Create 5 vectors all with similarity 0.7 to base
        # (using different orthogonal directions)
        for i in range(5):
            # Use different basis for orthogonal component
            orthogonal = create_basis_vector(DIMENSION, i + 1)
            vec = [
                0.7 * b + 0.714142842854285 * o  # sqrt(1-0.7^2) ≈ 0.714
                for b, o in zip(base, orthogonal, strict=False)
            ]
            # Normalize
            import math as m

            norm = m.sqrt(sum(x * x for x in vec))
            vec = [x / norm for x in vec]
            add_embedding(client, f"same_sim_{i}", vec)

        results = search(client, base, k=5, threshold=0.0)

        assert results["count"] == 5
        # All should have approximately the same score
        scores = [r["score"] for r in results["results"]]
        assert max(scores) - min(scores) < 0.05  # All within 0.05 of each other
