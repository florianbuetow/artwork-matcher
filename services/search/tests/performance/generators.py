"""
Embedding generators for performance testing.

Provides functions to generate test embeddings with controlled properties.
"""

from __future__ import annotations

import numpy as np


def create_normalized_embedding(dimension: int, seed: int | None = None) -> list[float]:
    """
    Generate a normalized random embedding vector.

    Normalized vectors ensure inner product equals cosine similarity,
    with scores in the range [-1, 1].

    Args:
        dimension: Vector dimension (must match FAISS index)
        seed: Random seed for reproducibility

    Returns:
        L2-normalized embedding as list of floats
    """
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(dimension).astype(np.float32)
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()
