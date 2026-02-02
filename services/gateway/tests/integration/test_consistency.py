"""
Integration tests for pipeline response consistency.

These tests verify that:
1. The same image always produces consistent identification results (determinism)
2. Different images may produce different results (discrimination capability)
3. Pipeline behavior is stable across multiple requests
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import httpx
import pytest
import respx
from PIL import Image

from tests.factories import create_mock_embedding

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.testclient import TestClient


# Backend service URLs (must match config.yaml)
EMBEDDINGS_URL = "http://localhost:8001"
SEARCH_URL = "http://localhost:8002"
GEOMETRIC_URL = "http://localhost:8003"


def create_colored_image(color: tuple[int, int, int], size: int = 100) -> str:
    """Create a solid color image and return as base64."""
    img = Image.new("RGB", (size, size), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def identify_artwork(client: TestClient, image_base64: str) -> dict:
    """Send identify request and return response data."""
    response = client.post("/identify", json={"image": image_base64})
    assert response.status_code == 200, f"Failed to identify: {response.text}"
    return response.json()


# =============================================================================
# Pipeline Response Consistency Tests
# =============================================================================


@pytest.mark.integration
class TestPipelineConsistency:
    """Tests verifying that pipeline responses are consistent."""

    @pytest.fixture
    def mock_consistent_pipeline(self) -> Iterator[respx.MockRouter]:
        """Mock pipeline with consistent responses."""
        with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
            # Mock embeddings
            router.get(f"{EMBEDDINGS_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{EMBEDDINGS_URL}/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": create_mock_embedding(768),
                        "dimension": 768,
                        "processing_time_ms": 50.0,
                        "image_id": None,
                    },
                )
            )

            # Mock search
            router.get(f"{SEARCH_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{SEARCH_URL}/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "object_id": "artwork_001",
                                "score": 0.92,
                                "rank": 1,
                                "metadata": {"name": "Water Lilies", "artist": "Monet"},
                            }
                        ],
                        "query_id": "test_query",
                        "k": 5,
                        "search_time_ms": 10.0,
                    },
                )
            )

            # Mock geometric
            router.get(f"{GEOMETRIC_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )

            yield router

    def test_same_image_produces_consistent_result(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_consistent_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Same image sent twice produces consistent match."""
        result1 = identify_artwork(client, sample_image_base64)
        result2 = identify_artwork(client, sample_image_base64)

        # Both should have successful results
        assert result1["success"] is True
        assert result2["success"] is True

        # Match should be the same object
        assert result1["match"]["object_id"] == result2["match"]["object_id"]
        assert result1["match"]["name"] == result2["match"]["name"]

    def test_match_confidence_is_consistent(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_consistent_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Confidence scores are consistent across requests."""
        result1 = identify_artwork(client, sample_image_base64)
        result2 = identify_artwork(client, sample_image_base64)

        # Confidence should be the same (within floating point tolerance)
        assert abs(result1["match"]["confidence"] - result2["match"]["confidence"]) < 0.001
        assert (
            abs(result1["match"]["similarity_score"] - result2["match"]["similarity_score"]) < 0.001
        )

    def test_consistency_across_multiple_requests(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_consistent_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Pipeline is stable across multiple requests."""
        results = [identify_artwork(client, sample_image_base64) for _ in range(5)]

        # All results should have the same object_id
        object_ids = [r["match"]["object_id"] for r in results]
        assert len(set(object_ids)) == 1, (
            f"Expected same object_id across all requests: {object_ids}"
        )

        # All results should report success
        assert all(r["success"] for r in results)


# =============================================================================
# Pipeline Discrimination Tests
# =============================================================================


@pytest.mark.integration
class TestPipelineDiscrimination:
    """Tests verifying that different inputs can produce different results."""

    @pytest.fixture
    def red_image_base64(self) -> str:
        """Solid red image."""
        return create_colored_image((255, 0, 0))

    @pytest.fixture
    def blue_image_base64(self) -> str:
        """Solid blue image."""
        return create_colored_image((0, 0, 255))

    @pytest.fixture
    def mock_red_pipeline(self) -> Iterator[respx.MockRouter]:
        """Mock pipeline for red image."""
        with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
            router.get(f"{EMBEDDINGS_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{EMBEDDINGS_URL}/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": [0.1] * 768,
                        "dimension": 768,
                        "processing_time_ms": 50.0,
                        "image_id": None,
                    },
                )
            )
            router.get(f"{SEARCH_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{SEARCH_URL}/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "object_id": "artwork_red",
                                "score": 0.95,
                                "rank": 1,
                                "metadata": {"name": "Red Composition"},
                            }
                        ],
                        "query_id": "test_query",
                        "k": 5,
                        "search_time_ms": 10.0,
                    },
                )
            )
            router.get(f"{GEOMETRIC_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            yield router

    @pytest.fixture
    def mock_blue_pipeline(self) -> Iterator[respx.MockRouter]:
        """Mock pipeline for blue image."""
        with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
            router.get(f"{EMBEDDINGS_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{EMBEDDINGS_URL}/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": [0.9] * 768,
                        "dimension": 768,
                        "processing_time_ms": 50.0,
                        "image_id": None,
                    },
                )
            )
            router.get(f"{SEARCH_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{SEARCH_URL}/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "object_id": "artwork_blue",
                                "score": 0.93,
                                "rank": 1,
                                "metadata": {"name": "Blue Study"},
                            }
                        ],
                        "query_id": "test_query",
                        "k": 5,
                        "search_time_ms": 10.0,
                    },
                )
            )
            router.get(f"{GEOMETRIC_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            yield router

    def test_different_pipelines_can_return_different_artworks(
        self,
        client: TestClient,
        red_image_base64: str,
        mock_red_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Pipeline can return different artworks based on backend response."""
        result_red = identify_artwork(client, red_image_base64)

        assert result_red["success"] is True
        assert result_red["match"]["object_id"] == "artwork_red"

    def test_blue_pipeline_returns_blue_artwork(
        self,
        client: TestClient,
        blue_image_base64: str,
        mock_blue_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Pipeline returns blue artwork when mocked accordingly."""
        result_blue = identify_artwork(client, blue_image_base64)

        assert result_blue["success"] is True
        assert result_blue["match"]["object_id"] == "artwork_blue"


# =============================================================================
# Response Structure Consistency Tests
# =============================================================================


@pytest.mark.integration
class TestResponseStructureConsistency:
    """Tests verifying response structure is consistent."""

    @pytest.fixture
    def mock_match_pipeline(self) -> Iterator[respx.MockRouter]:
        """Mock pipeline with a match."""
        with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
            router.get(f"{EMBEDDINGS_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{EMBEDDINGS_URL}/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": create_mock_embedding(768),
                        "dimension": 768,
                        "processing_time_ms": 50.0,
                        "image_id": None,
                    },
                )
            )
            router.get(f"{SEARCH_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{SEARCH_URL}/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "results": [
                            {"object_id": "art_001", "score": 0.9, "rank": 1, "metadata": {}}
                        ],
                        "query_id": "q",
                        "k": 5,
                        "search_time_ms": 5.0,
                    },
                )
            )
            router.get(f"{GEOMETRIC_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            yield router

    @pytest.fixture
    def mock_no_match_pipeline(self) -> Iterator[respx.MockRouter]:
        """Mock pipeline with no match."""
        with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
            router.get(f"{EMBEDDINGS_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{EMBEDDINGS_URL}/embed").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "embedding": create_mock_embedding(768),
                        "dimension": 768,
                        "processing_time_ms": 50.0,
                        "image_id": None,
                    },
                )
            )
            router.get(f"{SEARCH_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            router.post(f"{SEARCH_URL}/search").mock(
                return_value=httpx.Response(
                    200,
                    json={"results": [], "query_id": "q", "k": 5, "search_time_ms": 5.0},
                )
            )
            router.get(f"{GEOMETRIC_URL}/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            yield router

    def test_response_always_has_success_field(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Response always has success field."""
        result = identify_artwork(client, sample_image_base64)
        assert "success" in result

    def test_response_always_has_timing_field(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Response always has timing field."""
        result = identify_artwork(client, sample_image_base64)
        assert "timing" in result

    def test_timing_has_required_fields(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Timing always has required fields."""
        result = identify_artwork(client, sample_image_base64)
        timing = result["timing"]

        assert "embedding_ms" in timing
        assert "search_ms" in timing
        assert "geometric_ms" in timing
        assert "total_ms" in timing

    def test_timing_values_are_non_negative(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Timing values are non-negative."""
        result = identify_artwork(client, sample_image_base64)
        timing = result["timing"]

        assert timing["embedding_ms"] >= 0
        assert timing["search_ms"] >= 0
        assert timing["geometric_ms"] >= 0
        assert timing["total_ms"] >= 0

    def test_total_timing_is_at_least_component_sum(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """Total timing is at least the sum of components."""
        result = identify_artwork(client, sample_image_base64)
        timing = result["timing"]

        component_sum = timing["embedding_ms"] + timing["search_ms"] + timing["geometric_ms"]
        # Total should be >= sum (may include overhead)
        assert timing["total_ms"] >= component_sum * 0.9  # Allow 10% tolerance

    def test_no_match_response_has_null_match(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_no_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """No match response has null match field."""
        result = identify_artwork(client, sample_image_base64)
        assert result["match"] is None

    def test_no_match_response_still_succeeds(
        self,
        client: TestClient,
        sample_image_base64: str,
        mock_no_match_pipeline: respx.MockRouter,  # noqa: ARG002 - needed for side effects
    ) -> None:
        """No match is still a successful response (not an error)."""
        result = identify_artwork(client, sample_image_base64)
        assert result["success"] is True
