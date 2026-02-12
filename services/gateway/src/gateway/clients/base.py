"""
Base HTTP client for backend services.

Provides standardized error handling and request patterns.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
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

    def __init__(
        self,
        base_url: str,
        timeout: float,
        service_name: str,
        retry_max_attempts: int,
        retry_initial_backoff_seconds: float,
        retry_max_backoff_seconds: float,
        retry_jitter_seconds: float,
        circuit_breaker_failure_threshold: int,
        circuit_breaker_recovery_timeout_seconds: float,
    ) -> None:
        """
        Initialize the backend client.

        Args:
            base_url: Base URL for the service (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
            service_name: Name of the service for logging and error messages
            retry_max_attempts: Total attempts for transient request failures
            retry_initial_backoff_seconds: Initial backoff for retries
            retry_max_backoff_seconds: Maximum retry backoff cap
            retry_jitter_seconds: Random jitter added to backoff
            circuit_breaker_failure_threshold: Consecutive failures before opening circuit
            circuit_breaker_recovery_timeout_seconds: Cooldown before half-open probe
        """
        self.base_url = base_url
        self.service_name = service_name
        self.timeout = timeout

        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_backoff_seconds = retry_initial_backoff_seconds
        self.retry_max_backoff_seconds = retry_max_backoff_seconds
        self.retry_jitter_seconds = retry_jitter_seconds

        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout_seconds = circuit_breaker_recovery_timeout_seconds

        self._consecutive_failures = 0
        self._circuit_state = "closed"
        self._circuit_opened_at: float | None = None
        self._circuit_lock = asyncio.Lock()

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def _enforce_circuit_policy(self, path: str) -> None:
        """Fail fast when circuit is open, or transition to half-open after cooldown."""
        logger = get_logger()
        async with self._circuit_lock:
            if self._circuit_state != "open":
                return

            now = time.monotonic()
            if self._circuit_opened_at is None:
                logger.warning(
                    "Circuit opened without timestamp; repairing state",
                    extra={"backend": self.service_name, "path": path},
                )
                self._circuit_opened_at = now

            elapsed = now - self._circuit_opened_at
            if elapsed < self.circuit_breaker_recovery_timeout_seconds:
                raise BackendError(
                    error="backend_circuit_open",
                    message=f"{self.service_name} service temporarily unavailable",
                    status_code=503,
                    details={
                        "backend": self.service_name,
                        "path": path,
                        "recovery_timeout_seconds": self.circuit_breaker_recovery_timeout_seconds,
                        "failure_threshold": self.circuit_breaker_failure_threshold,
                    },
                )

            self._circuit_state = "half_open"

    async def _record_success(self, path: str) -> None:
        """Reset circuit failure state after a successful request."""
        logger = get_logger()
        async with self._circuit_lock:
            had_failures = self._consecutive_failures > 0
            was_open = self._circuit_state in {"open", "half_open"}

            self._consecutive_failures = 0
            self._circuit_state = "closed"
            self._circuit_opened_at = None

        if had_failures or was_open:
            logger.info(
                "Backend circuit closed after successful request",
                extra={"backend": self.service_name, "path": path},
            )

    async def _record_failure(self, path: str, reason: str) -> None:
        """Track request failures and open circuit when threshold is exceeded."""
        logger = get_logger()
        circuit_opened = False
        failure_count = 0

        async with self._circuit_lock:
            if self._circuit_state == "half_open":
                self._circuit_state = "open"
                self._circuit_opened_at = time.monotonic()
                self._consecutive_failures = self.circuit_breaker_failure_threshold
                circuit_opened = True
                failure_count = self._consecutive_failures
            elif self._circuit_state == "closed":
                self._consecutive_failures += 1
                failure_count = self._consecutive_failures

                if self._consecutive_failures >= self.circuit_breaker_failure_threshold:
                    self._circuit_state = "open"
                    self._circuit_opened_at = time.monotonic()
                    circuit_opened = True
            else:
                failure_count = self._consecutive_failures

        if circuit_opened:
            logger.warning(
                "Backend circuit opened",
                extra={
                    "backend": self.service_name,
                    "path": path,
                    "reason": reason,
                    "consecutive_failures": failure_count,
                    "failure_threshold": self.circuit_breaker_failure_threshold,
                    "recovery_timeout_seconds": self.circuit_breaker_recovery_timeout_seconds,
                },
            )

    def _is_retryable_status(self, status_code: int) -> bool:
        """Return whether an HTTP status should trigger retry."""
        return status_code in {408, 429, 500, 502, 503, 504}

    async def _should_retry(self, attempt: int, error: Exception) -> bool:
        """Return whether the request should be retried."""
        if attempt >= self.retry_max_attempts:
            return False

        async with self._circuit_lock:
            is_half_open = self._circuit_state == "half_open"

        if is_half_open:
            return False

        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
            return True

        if isinstance(error, httpx.HTTPStatusError):
            return self._is_retryable_status(error.response.status_code)

        return False

    def _compute_backoff_seconds(self, attempt: int) -> float:
        """Compute exponential backoff with jitter."""
        exponential = self.retry_initial_backoff_seconds * (2 ** (attempt - 1))
        capped_backoff = min(exponential, self.retry_max_backoff_seconds)

        jitter: float = 0.0
        if self.retry_jitter_seconds > 0:
            jitter = float(random.uniform(0.0, self.retry_jitter_seconds))  # nosec B311

        return float(capped_backoff + jitter)

    def _parse_backend_http_error(self, error: httpx.HTTPStatusError) -> tuple[str, str]:
        """Extract backend error code and message from HTTP status response."""
        error_code = "unknown"
        error_message = str(error)
        logger = get_logger()

        try:
            backend_error = error.response.json()
            # External API error response - fallback is intentional for malformed responses
            parsed_error_code = backend_error.get("error")  # nosemgrep: no-dict-get-with-default
            parsed_error_message = backend_error.get(
                "message"
            )  # nosemgrep: no-dict-get-with-default
            if parsed_error_code is not None:
                error_code = str(parsed_error_code)
            if parsed_error_message is not None:
                error_message = str(parsed_error_message)
        except json.JSONDecodeError:
            logger.warning(
                "Backend error payload was not valid JSON",
                extra={
                    "backend": self.service_name,
                    "status_code": error.response.status_code,
                    "path": str(error.request.url.path),
                },
            )

        return error_code, error_message

    async def health_check(self) -> str:
        """
        Check backend health.

        Returns:
            Status string ("healthy", "unhealthy", "unavailable", "error", or "unknown")
        """
        status = "unknown"
        logger = get_logger()

        try:
            await self._enforce_circuit_policy("/health")
            response = await self.client.get("/health")
            response.raise_for_status()
            data = response.json()
            await self._record_success("/health")

            # External API response - fallback is intentional for malformed responses
            parsed_status = data.get("status")  # nosemgrep: no-dict-get-with-default
            if parsed_status is not None:
                status = str(parsed_status)
        except BackendError:
            status = "unavailable"
            logger.warning(
                "Backend health unavailable due to open circuit",
                extra={"backend": self.service_name},
            )
        except httpx.ConnectError:
            await self._record_failure("/health", "connect_error")
            status = "unavailable"
            logger.warning(
                "Backend health check failed: connection error",
                extra={"backend": self.service_name},
            )
        except httpx.TimeoutException:
            await self._record_failure("/health", "timeout")
            status = "unavailable"
            logger.warning(
                "Backend health check failed: timeout",
                extra={"backend": self.service_name},
            )
        except httpx.HTTPStatusError as e:
            await self._record_failure("/health", "http_status_error")
            status = "unavailable" if e.response.status_code == 503 else "error"
            logger.warning(
                "Backend health check failed: HTTP status error",
                extra={
                    "backend": self.service_name,
                    "status_code": e.response.status_code,
                },
            )
        except json.JSONDecodeError:
            await self._record_failure("/health", "invalid_json")
            status = "error"
            logger.warning(
                "Backend health check failed: invalid JSON response",
                extra={"backend": self.service_name},
            )
        except Exception:
            await self._record_failure("/health", "unexpected_error")
            status = "error"
            logger.exception(
                "Backend health check failed: unexpected error",
                extra={"backend": self.service_name},
            )

        return status

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

        for attempt in range(1, self.retry_max_attempts + 1):
            await self._enforce_circuit_policy(path)

            try:
                response = await self.client.request(method, path, **kwargs)
                response.raise_for_status()
                response_data = response.json()
                await self._record_success(path)
                return dict(response_data)

            except json.JSONDecodeError as e:
                await self._record_failure(path, "invalid_json")
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

            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
                if await self._should_retry(attempt, e):
                    backoff_seconds = self._compute_backoff_seconds(attempt)
                    logger.warning(
                        "Backend request failed, retrying",
                        extra={
                            "backend": self.service_name,
                            "path": path,
                            "attempt": attempt,
                            "max_attempts": self.retry_max_attempts,
                            "backoff_seconds": round(backoff_seconds, 3),
                            "error_type": type(e).__name__,
                        },
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue

                if isinstance(e, httpx.TimeoutException):
                    await self._record_failure(path, "timeout")
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
                        details={
                            "backend": self.service_name,
                            "timeout_seconds": self.timeout,
                            "attempts": attempt,
                        },
                    ) from e

                if isinstance(e, httpx.ConnectError):
                    await self._record_failure(path, "connect_error")
                    logger.warning(
                        "Backend unavailable",
                        extra={"backend": self.service_name, "url": self.base_url},
                    )
                    raise BackendError(
                        error="backend_unavailable",
                        message=f"{self.service_name} service is not responding",
                        status_code=502,
                        details={
                            "backend": self.service_name,
                            "url": self.base_url,
                            "attempts": attempt,
                        },
                    ) from e

                await self._record_failure(path, "http_status_error")
                error_code, error_message = self._parse_backend_http_error(e)

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
                        "attempts": attempt,
                    },
                ) from e

        # Defensive fallback (should be unreachable because loop always returns or raises)
        raise BackendError(
            error="backend_error",
            message=f"{self.service_name} service request failed",
            status_code=502,
            details={"backend": self.service_name, "path": path},
        )
