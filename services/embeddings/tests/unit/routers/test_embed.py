"""
Unit tests for embed endpoint.

Tests the /embed endpoint with mocked model and dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embeddings_service.app import create_app
from tests.factories import (
    create_bmp_image_base64,
    create_invalid_base64,
    create_non_image_base64,
    create_random_embedding,
    create_test_image_base64,
)


@pytest.mark.unit
@pytest.mark.usefixtures("mock_settings")
class TestEmbedSuccess:
    """Tests for successful embedding extraction."""

    def test_embed_returns_200(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint returns 200 OK for valid image."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64()},
            )

            assert response.status_code == 200

    def test_embed_returns_embedding(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint returns embedding array."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64()},
            )
            data = response.json()

            assert "embedding" in data
            assert isinstance(data["embedding"], list)
            assert len(data["embedding"]) > 0

    def test_embed_returns_correct_dimension(self, mock_app_state: MagicMock) -> None:
        """Embedding has correct dimension (768 for dinov2-base)."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64()},
            )
            data = response.json()

            assert data["dimension"] == 768
            assert len(data["embedding"]) == 768

    def test_embed_returns_processing_time(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint returns processing time."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64()},
            )
            data = response.json()

            assert "processing_time_ms" in data
            assert isinstance(data["processing_time_ms"], (int, float))
            assert data["processing_time_ms"] >= 0

    def test_embed_echoes_image_id(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint echoes back image_id."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64(), "image_id": "test_001"},
            )
            data = response.json()

            assert data["image_id"] == "test_001"

    def test_embed_image_id_is_optional(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint works without image_id."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64()},
            )
            data = response.json()

            assert data["image_id"] is None

    def test_embed_embedding_is_l2_normalized(self, mock_app_state: MagicMock) -> None:
        """Embedding is L2-normalized (unit length)."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            # Return a normalized embedding
            embed_mock.return_value = np.array(
                create_random_embedding(768, normalized=True), dtype=np.float32
            )
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64()},
            )
            data = response.json()

            embedding = np.array(data["embedding"])
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"


@pytest.mark.unit
@pytest.mark.usefixtures("mock_settings")
class TestEmbedFormats:
    """Tests for different image formats."""

    def test_embed_accepts_jpeg(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint accepts JPEG images."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64(format="JPEG")},
            )

            assert response.status_code == 200

    def test_embed_accepts_png(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint accepts PNG images."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64(format="PNG")},
            )

            assert response.status_code == 200

    def test_embed_accepts_webp(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint accepts WebP images."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_test_image_base64(format="WEBP")},
            )

            assert response.status_code == 200

    def test_embed_rejects_bmp(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint rejects BMP format."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_bmp_image_base64()},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "unsupported_format"


@pytest.mark.unit
@pytest.mark.usefixtures("mock_settings")
class TestEmbedErrors:
    """Tests for error handling."""

    def test_embed_invalid_base64(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint rejects invalid base64."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_invalid_base64()},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "decode_error"

    def test_embed_not_an_image(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint rejects non-image data."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_non_image_base64()},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "invalid_image"

    def test_embed_missing_image_field(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint requires image field."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={},
            )

            assert response.status_code == 422  # Validation error

    def test_embed_unsupported_format_details(self, mock_app_state: MagicMock) -> None:
        """Unsupported format error includes details."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
        ):
            mock_state.return_value = mock_app_state
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/embed",
                json={"image": create_bmp_image_base64()},
            )

            data = response.json()
            assert "details" in data
            assert "detected_format" in data["details"]
            assert "supported_formats" in data["details"]


@pytest.mark.unit
@pytest.mark.usefixtures("mock_settings")
class TestEmbedDataUrl:
    """Tests for data URL handling."""

    def test_embed_accepts_data_url(self, mock_app_state: MagicMock) -> None:
        """Embed endpoint accepts data URL format."""
        with (
            patch("embeddings_service.app.lifespan"),
            patch("embeddings_service.routers.embed.get_app_state") as mock_state,
            patch("embeddings_service.routers.embed.get_logger"),
            patch("embeddings_service.routers.embed.extract_dino_embedding") as embed_mock,
        ):
            mock_state.return_value = mock_app_state
            embed_mock.return_value = np.array(create_random_embedding(768), dtype=np.float32)
            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            data_url = f"data:image/jpeg;base64,{create_test_image_base64()}"
            response = client.post(
                "/embed",
                json={"image": data_url},
            )

            assert response.status_code == 200
