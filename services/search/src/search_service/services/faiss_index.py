"""
FAISS index wrapper with metadata storage.

Provides a high-level interface for vector similarity search
with associated metadata for each stored embedding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import faiss
import numpy as np


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    object_id: str
    score: float
    rank: int
    metadata: dict[str, Any]


@dataclass
class IndexMetadataItem:
    """Metadata for a single indexed item."""

    index: int
    object_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FAISSIndexError(Exception):
    """Base exception for FAISS index operations."""

    pass


class DimensionMismatchError(FAISSIndexError):
    """Raised when embedding dimension doesn't match index dimension."""

    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"Embedding dimension {received} does not match index dimension {expected}"
        )


class InvalidEmbeddingError(FAISSIndexError):
    """Raised when embedding data is invalid."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid embedding: {reason}")


class IndexSaveError(FAISSIndexError):
    """Raised when index save operation fails."""

    pass


class IndexLoadError(FAISSIndexError):
    """Raised when index load operation fails."""

    pass


class FAISSIndex:
    """
    Wrapper around FAISS index with metadata storage.

    Manages a FAISS IndexFlatIP (inner product) index alongside
    a parallel list of metadata for each indexed item.

    Attributes:
        dimension: The embedding vector dimension
        index: The underlying FAISS index
        metadata_items: List of metadata for each indexed position
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize a new FAISS index.

        Args:
            dimension: Embedding vector dimension (must be > 0)

        Raises:
            ValueError: If dimension is not positive
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata_items: list[IndexMetadataItem] = []

    @property
    def count(self) -> int:
        """Get the number of vectors in the index."""
        return int(self.index.ntotal)

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self.count == 0

    def _validate_embedding(self, embedding: list[float]) -> np.ndarray:
        """
        Validate and convert embedding to numpy array.

        Args:
            embedding: Embedding vector as list of floats

        Returns:
            Validated numpy array

        Raises:
            DimensionMismatchError: If dimension doesn't match
            InvalidEmbeddingError: If embedding is malformed
        """
        if len(embedding) != self.dimension:
            raise DimensionMismatchError(
                expected=self.dimension,
                received=len(embedding),
            )

        try:
            arr = np.array(embedding, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise InvalidEmbeddingError(f"Cannot convert to float array: {e}") from e

        if not np.isfinite(arr).all():
            raise InvalidEmbeddingError("Embedding contains NaN or infinite values")

        return arr

    def add(
        self,
        object_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> int:
        """
        Add an embedding with metadata to the index.

        Args:
            object_id: Unique identifier for this object
            embedding: Embedding vector (must match index dimension)
            metadata: Optional metadata to store with this embedding

        Returns:
            Index position where the embedding was added

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match
            InvalidEmbeddingError: If embedding is malformed
        """
        arr = self._validate_embedding(embedding)

        # FAISS expects 2D array with shape (n_vectors, dimension)
        arr_2d = arr.reshape(1, -1)

        # Get position before adding
        position = self.count

        # Add to FAISS index
        add_fn: Any = self.index.add
        add_fn(arr_2d)

        # Store metadata
        if metadata is None:
            metadata = {}
        self.metadata_items.append(
            IndexMetadataItem(
                index=position,
                object_id=object_id,
                metadata=metadata,
            )
        )

        return position

    def search(
        self,
        embedding: list[float],
        k: int,
        threshold: float,
    ) -> list[SearchResult]:
        """
        Search for similar vectors in the index.

        Args:
            embedding: Query embedding vector
            k: Maximum number of results to return
            threshold: Minimum similarity score (0-1 for normalized vectors)

        Returns:
            List of SearchResult objects, ranked by score (highest first)

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match
            InvalidEmbeddingError: If embedding is malformed
        """
        if self.is_empty:
            return []

        arr = self._validate_embedding(embedding)
        arr_2d = arr.reshape(1, -1)

        # Limit k to actual index size
        effective_k = min(k, self.count)

        # Perform search
        search_fn: Any = self.index.search
        scores, indices = search_fn(arr_2d, effective_k)

        # Convert to results
        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True), start=1):
            # FAISS returns -1 for invalid indices (shouldn't happen with IndexFlatIP)
            if idx < 0:
                continue

            # Apply threshold filter
            if score < threshold:
                continue

            metadata_item = self.metadata_items[idx]
            results.append(
                SearchResult(
                    object_id=metadata_item.object_id,
                    score=float(score),
                    rank=rank,
                    metadata=metadata_item.metadata,
                )
            )

        return results

    def clear(self) -> int:
        """
        Clear all vectors and metadata from the index.

        Returns:
            Number of items that were removed
        """
        previous_count = self.count

        # Reset the FAISS index
        self.index.reset()

        # Clear metadata
        self.metadata_items.clear()

        return previous_count

    def save(self, index_path: Path, metadata_path: Path) -> int:
        """
        Persist index and metadata to disk.

        Args:
            index_path: Path to save FAISS index binary
            metadata_path: Path to save metadata JSON

        Returns:
            Number of bytes written for the index file

        Raises:
            IndexSaveError: If save operation fails
        """
        try:
            # Ensure parent directories exist
            index_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(index_path))

            # Save metadata as JSON
            metadata_dict = {
                "dimension": self.dimension,
                "count": self.count,
                "items": [
                    {
                        "index": item.index,
                        "object_id": item.object_id,
                        "metadata": item.metadata,
                    }
                    for item in self.metadata_items
                ],
            }

            with metadata_path.open("w") as f:
                json.dump(metadata_dict, f, indent=2)

            return index_path.stat().st_size

        except Exception as e:
            raise IndexSaveError(f"Failed to save index: {e}") from e

    def load(self, index_path: Path, metadata_path: Path) -> None:
        """
        Load index and metadata from disk.

        Args:
            index_path: Path to FAISS index binary
            metadata_path: Path to metadata JSON

        Raises:
            IndexLoadError: If load operation fails
            DimensionMismatchError: If loaded index dimension doesn't match
        """
        if not index_path.exists():
            raise IndexLoadError(f"Index file not found: {index_path}")

        if not metadata_path.exists():
            raise IndexLoadError(f"Metadata file not found: {metadata_path}")

        try:
            # Load FAISS index
            loaded_index = faiss.read_index(str(index_path))

            # Validate dimension
            if loaded_index.d != self.dimension:
                raise DimensionMismatchError(
                    expected=self.dimension,
                    received=loaded_index.d,
                )

            # Load metadata
            with metadata_path.open() as f:
                metadata_dict = json.load(f)

            # Validate required keys exist
            if "dimension" not in metadata_dict:
                raise IndexLoadError("Metadata missing required 'dimension' field")
            if "items" not in metadata_dict:
                raise IndexLoadError("Metadata missing required 'items' field")

            # Validate metadata dimension matches
            metadata_dimension = metadata_dict["dimension"]
            if metadata_dimension != self.dimension:
                raise DimensionMismatchError(
                    expected=self.dimension,
                    received=metadata_dimension,
                )

            # Parse metadata items
            items: list[IndexMetadataItem] = []
            for item_dict in metadata_dict["items"]:
                if "index" not in item_dict:
                    raise IndexLoadError("Item missing required 'index' field")
                if "object_id" not in item_dict:
                    raise IndexLoadError("Item missing required 'object_id' field")

                # metadata field is optional for backwards compatibility
                metadata: dict[str, Any] = {}
                if "metadata" in item_dict:
                    metadata = item_dict["metadata"]

                items.append(
                    IndexMetadataItem(
                        index=item_dict["index"],
                        object_id=item_dict["object_id"],
                        metadata=metadata,
                    )
                )

            # Validate counts match
            if loaded_index.ntotal != len(items):
                raise IndexLoadError(
                    f"Index count ({loaded_index.ntotal}) doesn't match "
                    f"metadata count ({len(items)})"
                )

            # Replace current index and metadata
            self.index = loaded_index
            self.metadata_items = items

        except (DimensionMismatchError, IndexLoadError):
            raise
        except Exception as e:
            raise IndexLoadError(f"Failed to load index: {e}") from e
