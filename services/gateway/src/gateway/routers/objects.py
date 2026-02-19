"""
Objects endpoints.

Provides endpoints to list and retrieve artwork metadata and images.
Images are served by proxying requests to the storage service.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from gateway.config import get_settings
from gateway.core.state import get_app_state
from gateway.logging import get_logger
from gateway.schemas import ObjectDetails, ObjectListResponse, ObjectSummary

router = APIRouter()


def load_metadata() -> dict[str, dict[str, Any]]:
    """
    Load metadata from labels.csv.

    Returns:
        Dictionary mapping object_id to metadata
    """
    settings = get_settings()
    labels_path = Path(settings.data.labels_path)

    if not labels_path.exists():
        return {}

    metadata: dict[str, dict[str, Any]] = {}

    try:
        with labels_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # External CSV data - handle multiple column naming conventions
                object_id = row.get("object_id")  # nosemgrep: no-dict-get-with-default
                if object_id is None:
                    object_id = row.get("id")  # nosemgrep: no-dict-get-with-default
                if object_id is None:
                    object_id = ""
                if object_id:
                    name = row.get("name")  # nosemgrep: no-dict-get-with-default
                    if name is None:
                        name = row.get("title")  # nosemgrep: no-dict-get-with-default
                    year = row.get("year")  # nosemgrep: no-dict-get-with-default
                    if year is None:
                        year = row.get("date")  # nosemgrep: no-dict-get-with-default
                    # nosemgrep: no-dict-get-with-default (optional CSV fields)
                    metadata[object_id] = {
                        "object_id": object_id,
                        "name": name,
                        "artist": row.get("artist"),
                        "year": year,
                        "description": row.get("description"),
                        "location": row.get("location"),
                    }
    except Exception as e:
        logger = get_logger()
        logger.error(f"Failed to load metadata: {e}", exc_info=True)

    return metadata


@router.get("/objects", response_model=ObjectListResponse)
async def list_objects() -> ObjectListResponse:
    """
    List all objects in the database.

    Returns:
        List of objects with basic metadata
    """
    metadata = load_metadata()

    objects = [
        ObjectSummary(
            object_id=obj["object_id"],
            # Internal metadata dict - fields may not exist
            name=obj.get("name"),  # nosemgrep: no-dict-get-with-default
            artist=obj.get("artist"),  # nosemgrep: no-dict-get-with-default
            year=obj.get("year"),  # nosemgrep: no-dict-get-with-default
        )
        for obj in metadata.values()
    ]

    # Sort by object_id
    objects.sort(key=lambda o: o.object_id)

    return ObjectListResponse(
        objects=objects,
        count=len(objects),
    )


@router.get("/objects/{object_id}", response_model=ObjectDetails)
async def get_object(object_id: str) -> ObjectDetails:
    """
    Get details for a specific object.

    Args:
        object_id: Object identifier

    Returns:
        Object details

    Raises:
        HTTPException: If object not found
    """
    metadata = load_metadata()

    if object_id not in metadata:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Object '{object_id}' not found in database",
                "details": {},
            },
        )

    obj = metadata[object_id]

    return ObjectDetails(
        object_id=obj["object_id"],
        # Internal metadata dict - fields may not exist
        name=obj.get("name"),  # nosemgrep: no-dict-get-with-default
        artist=obj.get("artist"),  # nosemgrep: no-dict-get-with-default
        year=obj.get("year"),  # nosemgrep: no-dict-get-with-default
        description=obj.get("description"),  # nosemgrep: no-dict-get-with-default
        location=obj.get("location"),  # nosemgrep: no-dict-get-with-default
        image_url=f"/objects/{object_id}/image",
    )


@router.get("/objects/{object_id}/image")
async def get_object_image(object_id: str) -> Response:
    """
    Get the reference image for an object.

    Proxies the request to the storage service.

    Args:
        object_id: Object identifier

    Returns:
        JPEG image bytes

    Raises:
        HTTPException: If object or image not found
    """
    state = get_app_state()
    image_bytes = await state.storage_client.get_image_bytes(object_id)

    if image_bytes is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "image_not_found",
                "message": f"Image for object '{object_id}' not found",
                "details": {},
            },
        )

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
    )
