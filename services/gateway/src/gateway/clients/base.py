"""
Base HTTP client for backend services.

Provides standardized error handling and request patterns.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from gateway.core.exceptions import BackendError
from gateway.logging import get_logger


class BackendClient:
    """
    Base class for backend service clients.

    Provides common functionality for HTTP communication with
    backend services (embeddings, search, geometric).
    """

    def __init__(self, base_url: str, timeout: float, service_name: str) -> None:
        """
        Initialize the backend client.

        Args:
            base_url: Base URL for the service (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
            service_name: Name of the service for logging and error messages
        """
        self.base_url = base_url
        self.service_name = service_name
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def health_check(self) -> str:
        """
        Check backend health.

        Returns:
            Status string ("healthy", "unhealthy", "unavailable", "error", or "unknown")
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            data = response.json()
            # External API response - fallback is intentional for malformed responses
            status = data.get("status")  # nosemgrep: no-dict-get-with-default
            return str(status) if status is not None else "unknown"
        except httpx.ConnectError:
            return "unavailable"
        except httpx.TimeoutException:
            return "unavailable"
        except httpx.HTTPStatusError as e:
            # 503 Service Unavailable means the backend is unavailable
            if e.response.status_code == 503:
                return "unavailable"
            return "error"
        except json.JSONDecodeError:
            return "error"

    async def get_info(self) -> dict[str, Any]:
        """
        Get service info.

        Returns:
            Service info dictionary
        """
        return await self._request("GET", "/info")

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make request with standardized error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            **kwargs: Additional arguments passed to httpx

        Returns:
            Response JSON as dictionary

        Raises:
            BackendError: If request fails
        """
        logger = get_logger()

        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return dict(response.json())

        except json.JSONDecodeError as e:
            logger.warning(
                "Backend returned invalid JSON",
                extra={
                    "backend": self.service_name,
                    "path": path,
                },
            )
            raise BackendError(
                error="invalid_response",
                message=f"{self.service_name} service returned invalid JSON response",
                status_code=502,
                details={"backend": self.service_name, "path": path},
            ) from e

        except httpx.TimeoutException as e:
            logger.warning(
                "Backend timeout",
                extra={
                    "backend": self.service_name,
                    "path": path,
                    "timeout": self.timeout,
                },
            )
            raise BackendError(
                error="backend_timeout",
                message=f"{self.service_name} service timed out",
                status_code=504,
                details={"backend": self.service_name, "timeout_seconds": self.timeout},
            ) from e

        except httpx.ConnectError as e:
            logger.warning(
                "Backend unavailable",
                extra={"backend": self.service_name, "url": self.base_url},
            )
            raise BackendError(
                error="backend_unavailable",
                message=f"{self.service_name} service is not responding",
                status_code=502,
                details={"backend": self.service_name, "url": self.base_url},
            ) from e

        except httpx.HTTPStatusError as e:
            error_code = "unknown"
            error_message = str(e)
            try:
                backend_error = e.response.json()
                # External API error response - fallback is intentional for malformed responses
                error_code = backend_error.get("error")  # nosemgrep: no-dict-get-with-default
                error_message = backend_error.get("message")  # nosemgrep: no-dict-get-with-default
                if error_code is None:
                    error_code = "unknown"
                if error_message is None:
                    error_message = str(e)
            except json.JSONDecodeError as parse_err:
                logger.debug(
                    "Failed to parse backend error response as JSON",
                    extra={
                        "backend": self.service_name,
                        "status_code": e.response.status_code,
                        "parse_error": str(parse_err),
                    },
                )
            except Exception as parse_err:
                logger.debug(
                    "Unexpected error parsing backend error response",
                    extra={
                        "backend": self.service_name,
                        "error_type": type(parse_err).__name__,
                    },
                )

            logger.warning(
                "Backend error",
                extra={
                    "backend": self.service_name,
                    "status_code": e.response.status_code,
                    "error": error_code,
                    "backend_message": error_message,
                },
            )
            raise BackendError(
                error="backend_error",
                message=f"{self.service_name} service error: {error_message}",
                status_code=502,
                details={
                    "backend": self.service_name,
                    "backend_error": error_code,
                    "backend_message": error_message,
                    "backend_status_code": e.response.status_code,
                },
            ) from e
