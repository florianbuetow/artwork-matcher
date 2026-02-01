"""
Integration tests for embedding consistency and discrimination.

These tests verify that:
1. The same image always produces identical embeddings (determinism)
2. Different images produce distinct embeddings (discrimination)
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def create_colored_image(color: tuple[int, int, int], size: int = 100) -> str:
    """Create a solid color image and return as base64."""
    img = Image.new("RGB", (size, size), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_embedding(client: TestClient, image_base64: str, image_id: str | None = None) -> np.ndarray:
    """Extract embedding from image via API."""
    payload = {"image": image_base64}
    if image_id:
        payload["image_id"] = image_id
    response = client.post("/embed", json=payload)
    assert response.status_code == 200, f"Failed to get embedding: {response.text}"
    return np.array(response.json()["embedding"])


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# =============================================================================
# Embedding Determinism Tests
# =============================================================================


@pytest.mark.integration
class TestEmbeddingDeterminism:
    """Tests verifying that embeddings are deterministic."""

    def test_same_image_produces_identical_vectors(
        self, client: TestClient, sample_image_base64: str
    ) -> None:
        """Same image sent twice produces identical embeddings."""
        emb1 = get_embedding(client, sample_image_base64, "run1")
        emb2 = get_embedding(client, sample_image_base64, "run2")

        # Vectors should be exactly identical
        assert np.allclose(emb1, emb2, rtol=1e-7, atol=1e-7), (
            f"Expected identical vectors, got max diff: {np.max(np.abs(emb1 - emb2))}"
        )

    def test_same_image_cosine_similarity_is_one(
        self, client: TestClient, sample_image_base64: str
    ) -> None:
        """Cosine similarity between same image embeddings is 1.0."""
        emb1 = get_embedding(client, sample_image_base64, "run1")
        emb2 = get_embedding(client, sample_image_base64, "run2")

        similarity = cosine_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 1e-6, f"Expected similarity ~1.0, got {similarity}"

    def test_determinism_across_multiple_requests(
        self, client: TestClient, sample_image_base64: str
    ) -> None:
        """Embedding remains stable across multiple requests."""
        embeddings = [get_embedding(client, sample_image_base64, f"run{i}") for i in range(5)]

        # All embeddings should be identical to the first one
        reference = embeddings[0]
        for i, emb in enumerate(embeddings[1:], start=2):
            assert np.allclose(reference, emb, rtol=1e-7, atol=1e-7), f"Run {i} differs from run 1"


# =============================================================================
# Embedding Discrimination Tests
# =============================================================================


@pytest.mark.integration
class TestEmbeddingDiscrimination:
    """Tests verifying that different images produce distinct embeddings."""

    @pytest.fixture
    def red_image_base64(self) -> str:
        """Solid red image."""
        return create_colored_image((255, 0, 0))

    @pytest.fixture
    def green_image_base64(self) -> str:
        """Solid green image."""
        return create_colored_image((0, 255, 0))

    @pytest.fixture
    def blue_image_base64(self) -> str:
        """Solid blue image."""
        return create_colored_image((0, 0, 255))

    def test_different_images_produce_different_vectors(
        self, client: TestClient, red_image_base64: str, blue_image_base64: str
    ) -> None:
        """Different images produce different embedding vectors."""
        emb_red = get_embedding(client, red_image_base64, "red")
        emb_blue = get_embedding(client, blue_image_base64, "blue")

        # Vectors should NOT be identical
        assert not np.allclose(emb_red, emb_blue, rtol=1e-3, atol=1e-3), (
            "Expected different vectors for different images"
        )

    def test_different_images_cosine_similarity_less_than_one(
        self, client: TestClient, red_image_base64: str, blue_image_base64: str
    ) -> None:
        """Cosine similarity between different images is less than 1.0."""
        emb_red = get_embedding(client, red_image_base64, "red")
        emb_blue = get_embedding(client, blue_image_base64, "blue")

        similarity = cosine_similarity(emb_red, emb_blue)
        assert similarity < 0.99, f"Expected similarity < 0.99, got {similarity}"

    def test_three_different_images_all_distinct(
        self,
        client: TestClient,
        red_image_base64: str,
        green_image_base64: str,
        blue_image_base64: str,
    ) -> None:
        """Three different images all have distinct embeddings."""
        emb_red = get_embedding(client, red_image_base64, "red")
        emb_green = get_embedding(client, green_image_base64, "green")
        emb_blue = get_embedding(client, blue_image_base64, "blue")

        # All pairwise similarities should be less than 1.0
        sim_rg = cosine_similarity(emb_red, emb_green)
        sim_rb = cosine_similarity(emb_red, emb_blue)
        sim_gb = cosine_similarity(emb_green, emb_blue)

        assert sim_rg < 0.99, f"Red-Green similarity too high: {sim_rg}"
        assert sim_rb < 0.99, f"Red-Blue similarity too high: {sim_rb}"
        assert sim_gb < 0.99, f"Green-Blue similarity too high: {sim_gb}"

    def test_embedding_captures_visual_difference(
        self, client: TestClient, red_image_base64: str, green_image_base64: str
    ) -> None:
        """Visually different images have measurably different embeddings."""
        emb_red = get_embedding(client, red_image_base64, "red")
        emb_green = get_embedding(client, green_image_base64, "green")

        # Calculate L2 distance
        l2_distance = np.linalg.norm(emb_red - emb_green)

        # For L2-normalized vectors, max distance is 2.0 (opposite directions)
        # Different colored images should have meaningful distance
        assert l2_distance > 0.1, f"Expected meaningful L2 distance, got {l2_distance}"
