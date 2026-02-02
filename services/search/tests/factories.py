"""
Test data factories for creating embeddings and other test data.

These factories provide consistent, reproducible test data generation
for both unit and integration tests.
"""

from __future__ import annotations

import math

import numpy as np


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


def create_invalid_embedding_nan(dimension: int) -> list[float]:
    """
    Create an embedding containing NaN values.

    Args:
        dimension: Vector dimension

    Returns:
        Embedding with NaN at positions 0, 50, and dimension//2
    """
    arr = np.ones(dimension, dtype=np.float32)
    # Insert NaN at multiple positions
    arr[0] = float("nan")
    if dimension > 50:
        arr[50] = float("nan")
    arr[dimension // 2] = float("nan")
    return arr.tolist()


def create_invalid_embedding_inf(dimension: int) -> list[float]:
    """
    Create an embedding containing infinity values.

    Args:
        dimension: Vector dimension

    Returns:
        Embedding with +inf and -inf values
    """
    arr = np.ones(dimension, dtype=np.float32)
    # Insert infinity at multiple positions
    arr[0] = float("inf")
    arr[1] = float("-inf")
    if dimension > 100:
        arr[100] = float("inf")
    return arr.tolist()


def create_wrong_dimension_embedding(target_dimension: int) -> list[float]:
    """
    Create an embedding with wrong dimension (one less than expected).

    Args:
        target_dimension: The correct dimension that the index expects

    Returns:
        Embedding with dimension = target_dimension - 1
    """
    wrong_dimension = target_dimension - 1
    return create_normalized_embedding(wrong_dimension, seed=999)


def create_zero_embedding(dimension: int) -> list[float]:
    """
    Create a zero embedding (all zeros).

    This is invalid for inner product search as it has no direction.

    Args:
        dimension: Vector dimension

    Returns:
        Embedding of all zeros
    """
    return [0.0] * dimension


def create_unnormalized_embedding(dimension: int, seed: int | None = None) -> list[float]:
    """
    Create an unnormalized embedding (not L2 normalized).

    For testing that the service handles unnormalized vectors correctly.

    Args:
        dimension: Vector dimension
        seed: Random seed for reproducibility

    Returns:
        Unnormalized embedding
    """
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randn(dimension).astype(np.float32)
    # Scale by a large factor to ensure it's clearly unnormalized
    arr = arr * 10.0
    return arr.tolist()


def validate_embedding_is_nan_free(embedding: list[float]) -> bool:
    """
    Check if an embedding contains any NaN values.

    Args:
        embedding: Embedding to check

    Returns:
        True if no NaN values, False otherwise
    """
    return not any(math.isnan(x) for x in embedding)


def validate_embedding_is_finite(embedding: list[float]) -> bool:
    """
    Check if an embedding contains only finite values.

    Args:
        embedding: Embedding to check

    Returns:
        True if all values are finite, False otherwise
    """
    return all(math.isfinite(x) for x in embedding)


# =============================================================================
# Deterministic Vector Factories
# =============================================================================
# These functions create vectors with known, predictable similarities
# for testing exact expected results.


def create_basis_vector(dimension: int, index: int) -> list[float]:
    """
    Create a basis vector (one-hot) at the specified index.

    The vector has 1.0 at position `index` and 0.0 elsewhere.
    Two different basis vectors are orthogonal (similarity = 0).

    Args:
        dimension: Vector dimension
        index: Position of the 1.0 value (0-indexed)

    Returns:
        Normalized basis vector

    Example:
        create_basis_vector(5, 0) -> [1.0, 0.0, 0.0, 0.0, 0.0]
        create_basis_vector(5, 2) -> [0.0, 0.0, 1.0, 0.0, 0.0]
    """
    if index >= dimension:
        msg = f"Index {index} out of bounds for dimension {dimension}"
        raise ValueError(msg)
    arr = np.zeros(dimension, dtype=np.float32)
    arr[index] = 1.0
    return arr.tolist()


def create_vector_with_known_similarity(base: list[float], target_similarity: float) -> list[float]:
    """
    Create a vector with a known inner product similarity to the base.

    Uses the formula: v = target_similarity * base + sqrt(1 - target_similarity^2) * orthogonal
    This creates a normalized vector at a specific angle to the base.

    Args:
        base: Normalized base vector
        target_similarity: Desired inner product similarity (-1.0 to 1.0)

    Returns:
        Normalized vector with the specified similarity to base

    Example:
        base = [1, 0, 0, ...]
        create_vector_with_known_similarity(base, 0.8) -> vector with <v, base> = 0.8
    """
    if not -1.0 <= target_similarity <= 1.0:
        msg = f"Target similarity must be in [-1, 1], got {target_similarity}"
        raise ValueError(msg)

    base_arr = np.array(base, dtype=np.float32)
    dimension = len(base)

    # Create an orthogonal vector by using a different basis direction
    # Find first non-zero index in base to create orthogonal
    orthogonal = np.zeros(dimension, dtype=np.float32)

    # Use Gram-Schmidt: pick a vector not parallel to base
    for i in range(dimension):
        candidate = np.zeros(dimension, dtype=np.float32)
        candidate[i] = 1.0
        # Project out the base component
        proj = np.dot(candidate, base_arr) * base_arr
        orthogonal = candidate - proj
        if np.linalg.norm(orthogonal) > 1e-6:
            orthogonal = orthogonal / np.linalg.norm(orthogonal)
            break

    # Create vector at target angle: cos(theta)*base + sin(theta)*orthogonal
    cos_theta = target_similarity
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

    result = cos_theta * base_arr + sin_theta * orthogonal
    result = result / np.linalg.norm(result)  # Ensure normalized

    return result.tolist()


def create_opposite_vector(base: list[float]) -> list[float]:
    """
    Create the opposite (negated) vector.

    The opposite vector has similarity = -1.0 to the base.

    Args:
        base: Base vector to negate

    Returns:
        Negated and normalized vector
    """
    arr = np.array(base, dtype=np.float32)
    arr = -arr
    arr = arr / np.linalg.norm(arr)
    return arr.tolist()


def create_scaled_similarity_vectors(
    dimension: int, similarities: list[float]
) -> tuple[list[float], list[tuple[float, list[float]]]]:
    """
    Create a base vector and multiple vectors with known similarities.

    This is useful for testing ranking: you provide the expected similarities
    in any order, and the function returns vectors that will rank in
    descending order of those similarities.

    Args:
        dimension: Vector dimension
        similarities: List of target similarities (each in [-1, 1])

    Returns:
        Tuple of (base_vector, [(similarity, vector), ...])
        The list is sorted by similarity descending for easy verification.

    Example:
        base, vectors = create_scaled_similarity_vectors(768, [0.9, 0.5, 0.3])
        # vectors[0] has similarity 0.9 to base (highest)
        # vectors[1] has similarity 0.5 to base
        # vectors[2] has similarity 0.3 to base (lowest)
    """
    # Create base as first basis vector
    base = create_basis_vector(dimension, 0)

    # Create vectors with known similarities
    result = []
    for sim in similarities:
        vec = create_vector_with_known_similarity(base, sim)
        result.append((sim, vec))

    # Sort by similarity descending
    result.sort(key=lambda x: x[0], reverse=True)

    return base, result


def compute_inner_product(v1: list[float], v2: list[float]) -> float:
    """
    Compute the inner product (dot product) of two vectors.

    For normalized vectors, this equals cosine similarity.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Inner product value
    """
    arr1 = np.array(v1, dtype=np.float32)
    arr2 = np.array(v2, dtype=np.float32)
    return float(np.dot(arr1, arr2))
