"""Tests for FAISS index wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from search_service.services.faiss_index import (
    DimensionMismatchError,
    FAISSIndex,
    IndexLoadError,
    InvalidEmbeddingError,
)


def create_normalized_embedding(dimension: int) -> list[float]:
    """Create a normalized random embedding."""
    arr = np.random.randn(dimension).astype(np.float32)
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()


class TestFAISSIndexInit:
    """Tests for FAISSIndex initialization."""

    def test_creates_empty_index(self) -> None:
        """New index starts empty."""
        index = FAISSIndex(dimension=768)

        assert index.count == 0
        assert index.is_empty is True
        assert index.dimension == 768

    def test_rejects_invalid_dimension(self) -> None:
        """Invalid dimensions are rejected."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            FAISSIndex(dimension=0)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            FAISSIndex(dimension=-1)


class TestFAISSIndexAdd:
    """Tests for adding embeddings to the index."""

    def test_adds_embedding(self) -> None:
        """Embedding is added successfully."""
        index = FAISSIndex(dimension=768)
        embedding = create_normalized_embedding(768)

        position = index.add(
            object_id="test_001",
            embedding=embedding,
            metadata={"name": "Test Object"},
        )

        assert position == 0
        assert index.count == 1
        assert index.is_empty is False

    def test_adds_multiple_embeddings(self) -> None:
        """Multiple embeddings are added with correct positions."""
        index = FAISSIndex(dimension=768)

        for i in range(5):
            embedding = create_normalized_embedding(768)
            position = index.add(
                object_id=f"test_{i:03d}",
                embedding=embedding,
                metadata={"index": i},
            )
            assert position == i

        assert index.count == 5

    def test_rejects_dimension_mismatch(self) -> None:
        """Mismatched dimension is rejected."""
        index = FAISSIndex(dimension=768)
        wrong_embedding = create_normalized_embedding(512)

        with pytest.raises(DimensionMismatchError) as exc_info:
            index.add(object_id="test", embedding=wrong_embedding, metadata=None)

        assert exc_info.value.expected == 768
        assert exc_info.value.received == 512

    def test_rejects_invalid_embedding(self) -> None:
        """Invalid embedding data is rejected."""
        index = FAISSIndex(dimension=3)

        with pytest.raises(InvalidEmbeddingError, match="NaN"):
            index.add(
                object_id="test",
                embedding=[1.0, float("nan"), 0.5],
                metadata=None,
            )

        with pytest.raises(InvalidEmbeddingError, match="infinite"):
            index.add(
                object_id="test",
                embedding=[1.0, float("inf"), 0.5],
                metadata=None,
            )


class TestFAISSIndexSearch:
    """Tests for searching the index."""

    def test_search_returns_results(self) -> None:
        """Search returns ranked results."""
        index = FAISSIndex(dimension=768)

        # Add some embeddings
        embeddings = [create_normalized_embedding(768) for _ in range(5)]
        for i, emb in enumerate(embeddings):
            index.add(object_id=f"obj_{i}", embedding=emb, metadata={"index": i})

        # Search with first embedding (should match itself best)
        # Use threshold=-1.0 to include all results regardless of score
        results = index.search(embedding=embeddings[0], k=3, threshold=-1.0)

        assert len(results) == 3
        assert results[0].rank == 1
        # Results are ranked by score descending
        assert results[0].score >= results[1].score
        assert results[1].score >= results[2].score

    def test_search_empty_index_returns_empty(self) -> None:
        """Searching empty index returns empty list."""
        index = FAISSIndex(dimension=768)
        embedding = create_normalized_embedding(768)

        results = index.search(embedding=embedding, k=5, threshold=0.0)

        assert results == []

    def test_search_respects_k_limit(self) -> None:
        """Search respects k parameter."""
        index = FAISSIndex(dimension=768)

        # Add 10 embeddings
        for i in range(10):
            index.add(
                object_id=f"obj_{i}",
                embedding=create_normalized_embedding(768),
                metadata=None,
            )

        query = create_normalized_embedding(768)
        results = index.search(embedding=query, k=3, threshold=0.0)

        assert len(results) <= 3

    def test_search_applies_threshold(self) -> None:
        """Search filters results below threshold."""
        index = FAISSIndex(dimension=768)

        # Add embeddings
        for i in range(5):
            index.add(
                object_id=f"obj_{i}",
                embedding=create_normalized_embedding(768),
                metadata=None,
            )

        query = create_normalized_embedding(768)

        # With very high threshold, should get fewer results
        results_high = index.search(embedding=query, k=10, threshold=0.99)
        results_low = index.search(embedding=query, k=10, threshold=0.0)

        assert len(results_high) <= len(results_low)

    def test_search_rejects_dimension_mismatch(self) -> None:
        """Search rejects mismatched dimension."""
        index = FAISSIndex(dimension=768)
        index.add(
            object_id="test",
            embedding=create_normalized_embedding(768),
            metadata=None,
        )

        wrong_query = create_normalized_embedding(512)

        with pytest.raises(DimensionMismatchError):
            index.search(embedding=wrong_query, k=5, threshold=0.0)


class TestFAISSIndexClear:
    """Tests for clearing the index."""

    def test_clear_removes_all(self) -> None:
        """Clear removes all vectors and metadata."""
        index = FAISSIndex(dimension=768)

        # Add some embeddings
        for i in range(5):
            index.add(
                object_id=f"obj_{i}",
                embedding=create_normalized_embedding(768),
                metadata=None,
            )

        assert index.count == 5

        removed = index.clear()

        assert removed == 5
        assert index.count == 0
        assert index.is_empty is True


class TestFAISSIndexPersistence:
    """Tests for saving and loading the index."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Index can be saved and loaded."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test.json"

        # Create and populate index
        index1 = FAISSIndex(dimension=768)
        for i in range(3):
            index1.add(
                object_id=f"obj_{i}",
                embedding=create_normalized_embedding(768),
                metadata={"name": f"Object {i}"},
            )

        # Save
        index1.save(index_path, metadata_path)

        assert index_path.exists()
        assert metadata_path.exists()

        # Load into new index
        index2 = FAISSIndex(dimension=768)
        index2.load(index_path, metadata_path)

        assert index2.count == 3
        assert index2.metadata_items[0].object_id == "obj_0"
        assert index2.metadata_items[0].metadata["name"] == "Object 0"

    def test_load_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Loading from missing file raises IndexLoadError."""
        index = FAISSIndex(dimension=768)
        missing_path = tmp_path / "nonexistent.index"
        metadata_path = tmp_path / "nonexistent.json"

        with pytest.raises(IndexLoadError, match="not found"):
            index.load(missing_path, metadata_path)

    def test_load_dimension_mismatch_raises_error(self, tmp_path: Path) -> None:
        """Loading index with wrong dimension raises error."""
        index_path = tmp_path / "test.index"
        metadata_path = tmp_path / "test.json"

        # Create index with dimension 512
        index_512 = FAISSIndex(dimension=512)
        index_512.add(
            object_id="test",
            embedding=create_normalized_embedding(512),
            metadata=None,
        )
        index_512.save(index_path, metadata_path)

        # Try to load into index with dimension 768
        index_768 = FAISSIndex(dimension=768)

        with pytest.raises(DimensionMismatchError):
            index_768.load(index_path, metadata_path)
