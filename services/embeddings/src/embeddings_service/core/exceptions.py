"""
Custom exception handlers for consistent error responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from embeddings_service.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI, Request


class ServiceError(Exception):
    """
    Base exception for service errors.

    Attributes:
        error: Machine-readable error code
        message: Human-readable description
        status_code: HTTP status code
        details: Additional context
    """

    def __init__(
        self,
        error: str,
        message: str,
        status_code: int,
        details: dict[str, object] | None,
    ) -> None:
        self.error = error
        self.message = message
        self.status_code = status_code
        if details is None:
            self.details: dict[str, object] = {}
        else:
            self.details = details
        super().__init__(message)


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
    app.add_exception_handler(
        ServiceError,
        # nosemgrep: no-type-ignore
        service_error_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)
