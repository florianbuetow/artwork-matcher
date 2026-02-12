"""Tests for BackendClient base class."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.core.exceptions import BackendError

if TYPE_CHECKING:
    from gateway.clients.base import BackendClient


@pytest.mark.unit
class TestBackendClientRequest:
    """Tests for BackendClient._request method."""

    async def test_request_timeout_raises_backend_error_504(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that timeout raises BackendError with 504 status."""
        mock_httpx_client.request.side_effect = httpx.TimeoutException("Connection timed out")

        with pytest.raises(BackendError) as exc_info:
            await backend_client._request("GET", "/test")

        assert exc_info.value.status_code == 504
        assert exc_info.value.error == "backend_timeout"
        assert "timed out" in exc_info.value.message
        assert mock_httpx_client.request.call_count == backend_client.retry_max_attempts

    async def test_request_connection_error_raises_backend_error_502(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that connection error raises BackendError with 502 status."""
        mock_httpx_client.request.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(BackendError) as exc_info:
            await backend_client._request("GET", "/test")

        assert exc_info.value.status_code == 502
        assert exc_info.value.error == "backend_unavailable"
        assert "not responding" in exc_info.value.message
        assert mock_httpx_client.request.call_count == backend_client.retry_max_attempts

    async def test_request_http_error_with_json_body_includes_details(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that HTTP error with JSON body includes error details."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "validation_error",
            "message": "Invalid input data",
        }

        http_error = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.request.side_effect = http_error

        with pytest.raises(BackendError) as exc_info:
            await backend_client._request("POST", "/test", json={"data": "test"})

        assert exc_info.value.status_code == 502
        assert exc_info.value.error == "backend_error"
        assert "validation_error" in str(exc_info.value.details)

    async def test_request_http_error_with_malformed_json_uses_fallback(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test that HTTP error with malformed JSON uses fallback message."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        http_error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.request.side_effect = http_error

        with pytest.raises(BackendError) as exc_info:
            await backend_client._request("GET", "/test")

        assert exc_info.value.status_code == 502
        assert exc_info.value.error == "backend_error"
        # Should contain the original error message as fallback
        assert "Internal Server Error" in exc_info.value.message
        assert mock_httpx_client.request.call_count == backend_client.retry_max_attempts

    async def test_request_success_returns_json(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test successful request returns JSON data."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value", "number": 42}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await backend_client._request("GET", "/test")

        assert result == {"key": "value", "number": 42}
        mock_httpx_client.request.assert_called_once_with("GET", "/test")

    async def test_circuit_breaker_opens_after_failure_threshold(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Repeated failures open the circuit and subsequent calls fail fast."""
        backend_client.circuit_breaker_failure_threshold = 2
        mock_httpx_client.request.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(BackendError):
            await backend_client._request("GET", "/test")

        with pytest.raises(BackendError):
            await backend_client._request("GET", "/test")

        with pytest.raises(BackendError) as exc_info:
            await backend_client._request("GET", "/test")

        assert exc_info.value.status_code == 503
        assert exc_info.value.error == "backend_circuit_open"

        # Third call should fail fast without making new HTTP attempts
        expected_calls = backend_client.retry_max_attempts * 2
        assert mock_httpx_client.request.call_count == expected_calls

    async def test_request_does_not_retry_non_transient_501(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """HTTP 501 is treated as non-retryable and should fail immediately."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 501
        mock_response.json.return_value = {"error": "not_implemented", "message": "Not implemented"}

        http_error = httpx.HTTPStatusError(
            "Not Implemented",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.request.side_effect = http_error

        with pytest.raises(BackendError):
            await backend_client._request("GET", "/test")

        assert mock_httpx_client.request.call_count == 1

    async def test_circuit_half_open_success_closes_circuit(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """A successful probe in half-open state should close the circuit."""
        backend_client.circuit_breaker_failure_threshold = 1
        backend_client.circuit_breaker_recovery_timeout_seconds = 0.0

        mock_httpx_client.request.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(BackendError):
            await backend_client._request("GET", "/test")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request.side_effect = None
        mock_httpx_client.request.return_value = mock_response

        result = await backend_client._request("GET", "/test")

        assert result == {"ok": True}
        assert backend_client._circuit_state == "closed"


@pytest.mark.unit
class TestBackendClientHealthCheck:
    """Tests for BackendClient.health_check method."""

    async def test_health_check_returns_healthy_on_success(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health check returns 'healthy' on successful response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.get.return_value = mock_response

        result = await backend_client.health_check()

        assert result == "healthy"
        mock_httpx_client.get.assert_called_once_with("/health")

    async def test_health_check_returns_unavailable_on_connect_error(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health check returns 'unavailable' on connection error."""
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection refused")

        result = await backend_client.health_check()

        assert result == "unavailable"

    async def test_health_check_returns_unavailable_on_timeout(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health check returns 'unavailable' on timeout."""
        mock_httpx_client.get.side_effect = httpx.TimeoutException("Timed out")

        result = await backend_client.health_check()

        assert result == "unavailable"

    async def test_health_check_returns_unknown_when_status_missing(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health check returns 'unknown' when status field is missing."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # No status field
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.get.return_value = mock_response

        result = await backend_client.health_check()

        assert result == "unknown"

    async def test_health_check_returns_unavailable_on_503(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health check returns 'unavailable' on 503 status."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503

        http_error = httpx.HTTPStatusError(
            "Service Unavailable",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.get.side_effect = http_error

        result = await backend_client.health_check()

        assert result == "unavailable"

    async def test_health_check_returns_error_on_other_http_error(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health check returns 'error' on non-503 HTTP errors."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        http_error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_httpx_client.get.side_effect = http_error

        result = await backend_client.health_check()

        assert result == "error"


@pytest.mark.unit
class TestBackendClientGetInfo:
    """Tests for BackendClient.get_info method."""

    async def test_get_info_returns_service_info(
        self,
        backend_client: BackendClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test get_info returns service information."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "test",
            "version": "1.0.0",
            "extra": "data",
        }
        mock_response.raise_for_status = MagicMock()

        mock_httpx_client.request.return_value = mock_response

        result = await backend_client.get_info()

        assert result == {"service": "test", "version": "1.0.0", "extra": "data"}
        mock_httpx_client.request.assert_called_once_with("GET", "/info")
