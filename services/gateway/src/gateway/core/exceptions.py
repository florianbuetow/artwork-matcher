"""
Custom exception handlers for consistent error responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi.responses import JSONResponse

from gateway.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from fastapi import FastAPI, Request
    from starlette.requests import Request as StarletteRequest

    # nosemgrep: no-module-level-constants (type alias for static type checking only)
    ExceptionHandler = Callable[
        [StarletteRequest, Exception],
        Coroutine[Any, Any, JSONResponse],
    ]


class ServiceError(Exception):
    """
    Base exception for service errors.

    Attributes:
        error: Machine-readable error code
        message: Human-readable description
        status_code: HTTP status code
        details: Additional context
    """

    # nosemgrep: no-default-parameter-values (optional details for exceptions)
    def __init__(
        self,
        error: str,
        message: str,
        status_code: int,
        details: dict[str, object] | None = None,
    ) -> None:
        self.error = error
        self.message = message
        self.status_code = status_code
        if details is None:
            self.details: dict[str, object] = {}
        else:
            self.details = details
        super().__init__(message)


class BackendError(ServiceError):
    """
    Exception for backend service errors.

    Used when a backend service (embeddings, search, geometric) returns
    an error, times out, or is unavailable.
    """

    # nosemgrep: no-default-parameter-values (optional details for exceptions)
    def __init__(
        self,
        error: str,
        message: str,
        status_code: int,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(error, message, status_code, details)


async def service_error_handler(
    request: Request,
    exc: ServiceError,
) -> JSONResponse:
    """Handle ServiceError exceptions."""
    logger = get_logger()
    logger.warning(
        "Service error",
        extra={
            "error_code": exc.error,
            "error_message": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "message": exc.message,
            "details": exc.details,
        },
    )


async def unhandled_exception_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Logs full traceback but returns sanitized error to client.
    """
    logger = get_logger()
    logger.exception(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "details": {},
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    # Cast to expected FastAPI handler type - our more specific signature is compatible
    app.add_exception_handler(
        ServiceError,
        cast("ExceptionHandler", service_error_handler),
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)
