"""Tests for identification endpoint."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import create_test_image_base64

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


@pytest.mark.unit
class TestIdentifyEndpoint:
    """Tests for POST /identify endpoint."""

    def test_identify_returns_match(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test successful artwork identification."""
        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["match"] is not None
        assert data["match"]["object_id"] == "object_001"
        assert "confidence" in data["match"]
        assert "timing" in data

    def test_identify_no_match(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test when no matching artwork is found."""
        # Return empty search results
        mock_app_state.search_client.search.return_value = []

        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["match"] is None
        assert "message" in data
        assert "timing" in data

    def test_identify_with_options(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test identification with custom options."""
        response = test_client.post(
            "/identify",
            json={
                "image": create_test_image_base64(),
                "options": {
                    "k": 10,
                    "threshold": 0.5,
                    "geometric_verification": False,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify search was called with custom k
        mock_app_state.search_client.search.assert_called()

    def test_identify_returns_timing_info(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test that timing information is included in response."""
        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 200
        data = response.json()
        assert "timing" in data
        timing = data["timing"]
        assert "embedding_ms" in timing
        assert "search_ms" in timing
        assert "geometric_ms" in timing
        assert "total_ms" in timing

    def test_identify_missing_image(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test error when image field is missing."""
        response = test_client.post(
            "/identify",
            json={},
        )

        assert response.status_code == 422  # Validation error

    def test_identify_empty_image(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,  # noqa: ARG002 - fixture required
    ) -> None:
        """Test error when image is empty string."""
        response = test_client.post(
            "/identify",
            json={"image": ""},
        )

        assert response.status_code == 422  # Validation error

    def test_identify_geometric_calls_match_batch_with_reference_images(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Geometric verification loads references and calls match_batch."""
        reference_path = tmp_path / "object_001.jpg"
        reference_bytes = b"fake-jpeg-bytes"
        reference_path.write_bytes(reference_bytes)

        batch_result = MagicMock()
        geo_result = MagicMock()
        geo_result.reference_id = "object_001"
        geo_result.confidence = 0.91
        batch_result.results = [geo_result]
        mock_app_state.geometric_client.match_batch.return_value = batch_result

        with patch("gateway.routers.identify.find_image_path", return_value=reference_path):
            response = test_client.post(
                "/identify",
                json={"image": create_test_image_base64()},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["match"]["object_id"] == "object_001"
        assert data["match"]["geometric_score"] == 0.91
        assert data["match"]["verification_method"] == "geometric"
        assert data["geometric_skipped"] is False

        mock_app_state.geometric_client.match_batch.assert_called_once()
        kwargs = mock_app_state.geometric_client.match_batch.call_args.kwargs
        assert kwargs["query_image"]
        assert kwargs["references"] == [
            {
                "reference_id": "object_001",
                "reference_image": base64.b64encode(reference_bytes).decode("ascii"),
            }
        ]

    def test_identify_geometric_falls_back_when_references_missing(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Missing references should fall back to embedding-only scoring."""
        with patch("gateway.routers.identify.find_image_path", return_value=None):
            response = test_client.post(
                "/identify",
                json={"image": create_test_image_base64()},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["match"]["verification_method"] == "embedding_only"
        assert data["match"]["geometric_score"] is None
        assert data["geometric_skipped"] is True
        assert data["geometric_skip_reason"] == "no_reference_images"
        mock_app_state.geometric_client.match_batch.assert_not_called()

    def test_identify_geometric_falls_back_on_backend_error(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Geometric backend errors should degrade gracefully to embedding-only."""
        from gateway.core.exceptions import BackendError  # noqa: PLC0415

        reference_path = tmp_path / "object_001.jpg"
        reference_path.write_bytes(b"fake-jpeg-bytes")
        mock_app_state.geometric_client.match_batch.side_effect = BackendError(
            error="geometric_error",
            message="Geometric service unavailable",
            status_code=502,
            details={"service": "geometric"},
        )

        with patch("gateway.routers.identify.find_image_path", return_value=reference_path):
            response = test_client.post(
                "/identify",
                json={"image": create_test_image_base64()},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["match"]["verification_method"] == "embedding_only"
        assert data["match"]["geometric_score"] is None
        assert data["geometric_skipped"] is True
        assert data["geometric_skip_reason"] == "backend_error"

    def test_identify_geometric_skips_candidate_when_reference_read_fails(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """OSError while loading references should not fail the request."""
        bad_path = MagicMock()
        bad_path.read_bytes.side_effect = OSError("read failed")

        with patch("gateway.routers.identify.find_image_path", return_value=bad_path):
            response = test_client.post(
                "/identify",
                json={"image": create_test_image_base64()},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["match"]["verification_method"] == "embedding_only"
        assert data["match"]["geometric_score"] is None
        assert data["geometric_skipped"] is True
        assert data["geometric_skip_reason"] == "no_reference_images"
        mock_app_state.geometric_client.match_batch.assert_not_called()

    def test_identify_geometric_marks_no_results_when_batch_empty(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Empty batch response should set no_results skip reason."""
        reference_path = tmp_path / "object_001.jpg"
        reference_path.write_bytes(b"fake-jpeg-bytes")

        batch_result = MagicMock()
        batch_result.results = []
        mock_app_state.geometric_client.match_batch.return_value = batch_result

        with patch("gateway.routers.identify.find_image_path", return_value=reference_path):
            response = test_client.post(
                "/identify",
                json={"image": create_test_image_base64()},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["match"]["verification_method"] == "embedding_only"
        assert data["match"]["geometric_score"] is None
        assert data["geometric_skipped"] is True
        assert data["geometric_skip_reason"] == "no_results"


@pytest.mark.unit
class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_confidence_with_geometric_high_score(self) -> None:
        """Test confidence calculation when geometric verification passes."""
        from gateway.routers.identify import calculate_confidence  # noqa: PLC0415

        # High similarity, high geometric score
        confidence = calculate_confidence(0.9, 0.8, geometric_enabled=True)
        # 0.6 * 0.9 + 0.4 * 0.8 = 0.54 + 0.32 = 0.86
        assert 0.85 <= confidence <= 0.87

    def test_confidence_with_geometric_low_score(self) -> None:
        """Test confidence calculation when geometric verification fails."""
        from gateway.routers.identify import calculate_confidence  # noqa: PLC0415

        # High similarity but low geometric score
        confidence = calculate_confidence(0.9, 0.3, geometric_enabled=True)
        # 0.3 * 0.9 + 0.2 * 0.3 = 0.27 + 0.06 = 0.33
        assert 0.32 <= confidence <= 0.34

    def test_confidence_without_geometric_enabled(self) -> None:
        """Test confidence when geometric was supposed to run but didn't."""
        from gateway.routers.identify import calculate_confidence  # noqa: PLC0415

        confidence = calculate_confidence(0.9, None, geometric_enabled=True)
        # 0.9 * 0.7 = 0.63
        assert 0.62 <= confidence <= 0.64

    def test_confidence_without_geometric_disabled(self) -> None:
        """Test confidence when geometric was intentionally skipped."""
        from gateway.routers.identify import calculate_confidence  # noqa: PLC0415

        confidence = calculate_confidence(0.9, None, geometric_enabled=False)
        # 0.9 * 0.85 = 0.765
        assert 0.76 <= confidence <= 0.77


@pytest.mark.unit
class TestIdentifyBackendErrors:
    """Tests for backend error handling in /identify endpoint."""

    def test_identify_embeddings_backend_error(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test that embeddings service errors return 502 with proper error response."""
        from gateway.core.exceptions import BackendError  # noqa: PLC0415

        mock_app_state.embeddings_client.embed.side_effect = BackendError(
            error="embeddings_error",
            message="Embeddings service unavailable",
            status_code=502,
            details={"service": "embeddings"},
        )

        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "embeddings_error"
        assert "embeddings" in data["message"].lower()

    def test_identify_search_backend_error(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test that search service errors return 502 with proper error response."""
        from gateway.core.exceptions import BackendError  # noqa: PLC0415

        mock_app_state.search_client.search.side_effect = BackendError(
            error="search_error",
            message="Search service unavailable",
            status_code=502,
            details={"service": "search"},
        )

        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "search_error"
        assert "search" in data["message"].lower()

    def test_identify_embeddings_timeout(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test that embeddings service timeout returns 504."""
        from gateway.core.exceptions import BackendError  # noqa: PLC0415

        mock_app_state.embeddings_client.embed.side_effect = BackendError(
            error="timeout",
            message="Embeddings service timed out",
            status_code=504,
            details={"service": "embeddings", "timeout_seconds": 30},
        )

        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 504
        data = response.json()
        assert data["error"] == "timeout"

    def test_identify_search_timeout(
        self,
        test_client: TestClient,
        mock_app_state: MagicMock,
    ) -> None:
        """Test that search service timeout returns 504."""
        from gateway.core.exceptions import BackendError  # noqa: PLC0415

        mock_app_state.search_client.search.side_effect = BackendError(
            error="timeout",
            message="Search service timed out",
            status_code=504,
            details={"service": "search", "timeout_seconds": 30},
        )

        response = test_client.post(
            "/identify",
            json={"image": create_test_image_base64()},
        )

        assert response.status_code == 504
        data = response.json()
        assert data["error"] == "timeout"
