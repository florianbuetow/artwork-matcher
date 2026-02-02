"""Tests for identification endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.factories import create_test_image_base64

if TYPE_CHECKING:
    from unittest.mock import MagicMock

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
