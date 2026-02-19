"""
Object storage endpoints.

Provides PUT/GET/DELETE operations for binary objects.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, Response

from storage_service.config import get_settings
from storage_service.core.exceptions import ServiceError
from storage_service.core.state import get_app_state
from storage_service.schemas import DeleteAllResponse

if TYPE_CHECKING:
    from storage_service.services.blob_store import FileBlobStore

router = APIRouter()

_VALID_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_object_id(object_id: str) -> None:
    """Validate object ID format. Raises ServiceError on invalid ID."""
    if not _VALID_ID_PATTERN.match(object_id):
        raise ServiceError(
            error="invalid_id",
            message=(
                f"Invalid object ID: '{object_id}'. IDs must contain only "
                "alphanumeric characters, hyphens, and underscores."
            ),
            status_code=400,
            details={"object_id": object_id},
        )


def _get_blob_store() -> FileBlobStore:
    """Get initialized blob store or raise a service error."""
    state = get_app_state()
    store = state.blob_store
    if store is None:
        raise ServiceError(
            error="service_unavailable",
            message="Blob store is not initialized",
            status_code=503,
            details={},
        )
    return store


@router.put("/objects/{object_id}", status_code=204)
async def put_object(object_id: str, request: Request) -> Response:
    """Store a binary object."""
    _validate_object_id(object_id)
    store = _get_blob_store()
    data = await request.body()
    try:
        store.put(object_id, data)
    except OSError as exc:
        raise ServiceError(
            error="storage_write_failed",
            message=f"Failed to store object '{object_id}'",
            status_code=503,
            details={
                "object_id": object_id,
                "exception_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        ) from exc
    return Response(status_code=204)


@router.get("/objects/{object_id}")
async def get_object(object_id: str) -> Response:
    """Retrieve a binary object."""
    _validate_object_id(object_id)
    settings = get_settings()
    store = _get_blob_store()
    try:
        data = store.get(object_id)
    except OSError as exc:
        raise ServiceError(
            error="storage_read_failed",
            message=f"Failed to read object '{object_id}'",
            status_code=503,
            details={
                "object_id": object_id,
                "exception_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        ) from exc
    if data is None:
        raise ServiceError(
            error="not_found",
            message=f"Object '{object_id}' not found",
            status_code=404,
            details={"object_id": object_id},
        )
    return Response(content=data, media_type=settings.storage.content_type)


@router.delete("/objects/{object_id}", status_code=204)
async def delete_object(object_id: str) -> Response:
    """Delete a single object."""
    _validate_object_id(object_id)
    store = _get_blob_store()
    try:
        deleted = store.delete(object_id)
    except OSError as exc:
        raise ServiceError(
            error="storage_delete_failed",
            message=f"Failed to delete object '{object_id}'",
            status_code=503,
            details={
                "object_id": object_id,
                "exception_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        ) from exc
    if not deleted:
        raise ServiceError(
            error="not_found",
            message=f"Object '{object_id}' not found",
            status_code=404,
            details={"object_id": object_id},
        )
    return Response(status_code=204)


@router.delete("/objects", response_model=DeleteAllResponse)
async def delete_all_objects() -> DeleteAllResponse:
    """Delete all stored objects."""
    store = _get_blob_store()
    try:
        count = store.delete_all()
    except OSError as exc:
        raise ServiceError(
            error="storage_delete_all_failed",
            message="Failed to delete all stored objects",
            status_code=503,
            details={
                "exception_type": exc.__class__.__name__,
                "reason": str(exc),
            },
        ) from exc
    return DeleteAllResponse(deleted_count=count)
