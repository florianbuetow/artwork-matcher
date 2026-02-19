"""Tests for StorageClient."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.core.exceptions import BackendError

if TYPE_CHECKING:
    from gateway.clients.storage import StorageClient


@pytest.mark.unit
class TestStorageClientGetImageBytes:
    """Tests for StorageClient.get_image_bytes method."""

    async def test_get_image_bytes_success_returns_bytes(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful image retrieval returns raw bytes."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b"image-bytes"
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.get.return_value = mock_response

        result = await storage_client.get_image_bytes("obj_001")

        assert result == b"image-bytes"
        mock_httpx_client.get.assert_called_once_with("/objects/obj_001")

    async def test_get_image_bytes_not_found_returns_none(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test 404 response returns None instead of raising an error."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404

        mock_httpx_client.get.return_value = mock_response

        result = await storage_client.get_image_bytes("obj_404")

        assert result is None
        mock_httpx_client.get.assert_called_once_with("/objects/obj_404")

    async def test_get_image_bytes_records_success_on_404(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test 404 is treated as successful request for circuit breaker metrics."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_httpx_client.get.return_value = mock_response

        record_success_spy = AsyncMock(wraps=storage_client._record_success)
        storage_client._record_success = record_success_spy

        result = await storage_client.get_image_bytes("obj_404")

        assert result is None
        record_success_spy.assert_awaited_once_with("/objects/obj_404")

    async def test_get_image_bytes_timeout_raises_backend_error_504(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test timeout errors map to BackendError with 504 status."""
        mock_httpx_client.get.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(BackendError) as exc_info:
            await storage_client.get_image_bytes("obj_timeout")

        assert exc_info.value.status_code == 504
        assert exc_info.value.error == "backend_timeout"

    async def test_get_image_bytes_connect_error_raises_backend_error_502(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test connection errors map to BackendError with 502 status."""
        mock_httpx_client.get.side_effect = httpx.ConnectError("connect error")

        with pytest.raises(BackendError) as exc_info:
            await storage_client.get_image_bytes("obj_connect")

        assert exc_info.value.status_code == 502
        assert exc_info.value.error == "backend_unavailable"

    async def test_get_image_bytes_http_error_raises_backend_error_502(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test HTTP status errors map to BackendError with 502 status."""
        request = httpx.Request("GET", "http://test")
        response = httpx.Response(500, request=request)
        mock_httpx_client.get.side_effect = httpx.HTTPStatusError(
            message="error",
            request=request,
            response=response,
        )

        with pytest.raises(BackendError) as exc_info:
            await storage_client.get_image_bytes("obj_error")

        assert exc_info.value.status_code == 502
        assert exc_info.value.error == "backend_error"

    async def test_get_image_bytes_includes_object_id_in_error_details(
        self,
        storage_client: StorageClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test BackendError details include object_id for troubleshooting."""
        mock_httpx_client.get.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(BackendError) as exc_info:
            await storage_client.get_image_bytes("obj_123")

        assert "object_id" in exc_info.value.details
        assert exc_info.value.details["object_id"] == "obj_123"


@pytest.mark.unit
class TestStorageClientGetImageBase64:
    """Tests for StorageClient.get_image_base64 method."""

    async def test_get_image_base64_success_returns_encoded_string(
        self,
        storage_client: StorageClient,
    ) -> None:
        """Test successful image retrieval returns base64-encoded string."""
        raw_bytes = b"test data"
        storage_client.get_image_bytes = AsyncMock(return_value=raw_bytes)

        result = await storage_client.get_image_base64("obj_001")

        assert result == base64.b64encode(raw_bytes).decode("ascii")
        storage_client.get_image_bytes.assert_awaited_once_with("obj_001")

    async def test_get_image_base64_not_found_returns_none(
        self,
        storage_client: StorageClient,
    ) -> None:
        """Test None from get_image_bytes propagates as None."""
        storage_client.get_image_bytes = AsyncMock(return_value=None)

        result = await storage_client.get_image_base64("obj_missing")

        assert result is None
        storage_client.get_image_bytes.assert_awaited_once_with("obj_missing")

    async def test_get_image_base64_propagates_backend_error(
        self,
        storage_client: StorageClient,
    ) -> None:
        """Test BackendError from get_image_bytes is propagated unchanged."""
        backend_error = BackendError(
            error="backend_timeout",
            message="storage service timed out",
            status_code=504,
            details={"backend": "storage", "object_id": "obj_001"},
        )
        storage_client.get_image_bytes = AsyncMock(side_effect=backend_error)

        with pytest.raises(BackendError) as exc_info:
            await storage_client.get_image_base64("obj_001")

        assert exc_info.value is backend_error
