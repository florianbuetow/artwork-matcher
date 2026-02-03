"""Tests for EmbeddingsClient."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.core.exceptions import BackendError

if TYPE_CHECKING:
    from gateway.clients.embeddings import EmbeddingsClient


@pytest.mark.unit
class TestEmbeddingsClientEmbed:
    """Tests for EmbeddingsClient.embed method."""

    async def test_embed_success_returns_embedding_list(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful embedding extraction returns list of floats."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "dimension": 5,
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await embeddings_client.embed("base64_image_data")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert all(isinstance(x, float) for x in result)

    async def test_embed_with_image_id(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test embedding extraction with optional image_id."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        await embeddings_client.embed("base64_image_data", image_id="test_image_001")

        # Verify the request was made with image_id
        call_kwargs = mock_httpx_client.request.call_args
        assert call_kwargs[1]["json"]["image_id"] == "test_image_001"

    async def test_embed_missing_embedding_field_raises_backend_error(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that missing embedding field raises BackendError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}  # No embedding field
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(BackendError) as exc_info:
            await embeddings_client.embed("base64_image_data")

        assert exc_info.value.error == "invalid_response"
        assert "embedding" in exc_info.value.message
        assert exc_info.value.status_code == 502

    async def test_embed_non_list_embedding_raises_backend_error(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that non-list embedding raises BackendError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": "not_a_list"}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(BackendError) as exc_info:
            await embeddings_client.embed("base64_image_data")

        assert exc_info.value.error == "invalid_response"
        assert "non-list" in exc_info.value.message
        assert exc_info.value.status_code == 502

    async def test_embed_empty_embedding_raises_backend_error(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that empty embedding raises BackendError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": []}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(BackendError) as exc_info:
            await embeddings_client.embed("base64_image_data")

        assert exc_info.value.error == "empty_embedding"
        assert "empty" in exc_info.value.message
        assert exc_info.value.status_code == 502

    async def test_embed_non_numeric_values_raises_backend_error(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that non-numeric values in embedding raises BackendError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, "not_a_number", 0.3]}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(BackendError) as exc_info:
            await embeddings_client.embed("base64_image_data")

        assert exc_info.value.error == "invalid_response"
        assert "non-numeric" in exc_info.value.message
        assert exc_info.value.status_code == 502


@pytest.mark.unit
class TestEmbeddingsClientGetEmbeddingDimension:
    """Tests for EmbeddingsClient.get_embedding_dimension method."""

    async def test_get_embedding_dimension_success(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful retrieval of embedding dimension."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "embeddings",
            "model": {
                "name": "dinov2",
                "embedding_dimension": 768,
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await embeddings_client.get_embedding_dimension()

        assert result == 768

    async def test_get_embedding_dimension_missing_model_info_raises_key_error(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that missing model info raises KeyError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"service": "embeddings"}  # No model field
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(KeyError) as exc_info:
            await embeddings_client.get_embedding_dimension()

        assert "model info" in str(exc_info.value)

    async def test_get_embedding_dimension_missing_dimension_raises_key_error(
        self,
        embeddings_client: EmbeddingsClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that missing embedding_dimension raises KeyError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "embeddings",
            "model": {"name": "dinov2"},  # No embedding_dimension
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(KeyError) as exc_info:
            await embeddings_client.get_embedding_dimension()

        assert "embedding_dimension" in str(exc_info.value)
