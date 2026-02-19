"""Custom exception handlers for consistent error responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from service_commons.exceptions import ServiceError
from service_commons.exceptions import (
    register_exception_handlers as register_common_exception_handlers,
)

from storage_service.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

__all__ = [
    "ServiceError",
    "register_exception_handlers",
    "service_error_handler",
    "unhandled_exception_handler",
]


async def service_error_handler(
    request: Request,
    exc: ServiceError,
) -> JSONResponse:
    """Handle ServiceError exceptions."""
    logger = get_logger(__name__)
    logger.warning(
        "Service error",
        extra={
            "error_code": exc.error,
            "error_message": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": str(request.url.path),
            "method": request.method,
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
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger = get_logger(__name__)
    logger.exception(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
        },
    )
    exception_message = str(exc)
    message = (
        exception_message
        if exception_message
        else f"Unhandled exception of type {exc.__class__.__name__}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": message,
            "details": {
                "exception_type": exc.__class__.__name__,
                "path": str(request.url.path),
                "method": request.method,
            },
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    register_common_exception_handlers(
        app,
        ServiceError,
        service_error_handler,
        unhandled_exception_handler,
    )
