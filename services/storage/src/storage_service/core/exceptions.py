"""Custom exception handlers for consistent error responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from service_commons.exceptions import (
    ServiceError,
    create_exception_handlers,
)
from service_commons.exceptions import (
    register_exception_handlers as register_common_exception_handlers,
)

from storage_service.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

__all__ = [
    "ServiceError",
    "register_exception_handlers",
    "service_error_handler",
    "unhandled_exception_handler",
]


_base_service_error_handler, _base_unhandled_exception_handler = create_exception_handlers(
    lambda: get_logger(__name__)
)


async def service_error_handler(
    request: Request,
    exc: ServiceError,
) -> JSONResponse:
    """Handle ServiceError exceptions."""
    return await _base_service_error_handler(request, exc)


async def unhandled_exception_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""
    return await _base_unhandled_exception_handler(request, _exc)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    register_common_exception_handlers(
        app,
        ServiceError,
        service_error_handler,
        unhandled_exception_handler,
    )
