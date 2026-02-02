"""
Objects endpoints.

Provides endpoints to list and retrieve artwork metadata and images.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from gateway.config import get_settings
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
        logger.warning(f"Failed to load metadata: {e}")

    return metadata


def find_image_path(object_id: str) -> Path | None:
    """
    Find the image file for an object.

    Searches for common image extensions.

    Returns:
        Path to image file or None if not found
    """
    settings = get_settings()
    objects_path = Path(settings.data.objects_path)

    extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in extensions:
        # Try object_id directly
        path = objects_path / f"{object_id}{ext}"
        if path.exists():
            return path

        # Try in subdirectory
        path = objects_path / object_id / f"image{ext}"
        if path.exists():
            return path

    return None


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
            },
        )

    obj = metadata[object_id]

    # Check if image exists
    image_path = find_image_path(object_id)
    image_url = f"/objects/{object_id}/image" if image_path else None

    return ObjectDetails(
        object_id=obj["object_id"],
        # Internal metadata dict - fields may not exist
        name=obj.get("name"),  # nosemgrep: no-dict-get-with-default
        artist=obj.get("artist"),  # nosemgrep: no-dict-get-with-default
        year=obj.get("year"),  # nosemgrep: no-dict-get-with-default
        description=obj.get("description"),  # nosemgrep: no-dict-get-with-default
        location=obj.get("location"),  # nosemgrep: no-dict-get-with-default
        image_url=image_url,
    )


@router.get("/objects/{object_id}/image")
async def get_object_image(object_id: str) -> FileResponse:
    """
    Get the reference image for an object.

    Args:
        object_id: Object identifier

    Returns:
        Image file

    Raises:
        HTTPException: If object or image not found
    """
    metadata = load_metadata()

    if object_id not in metadata:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Object '{object_id}' not found in database",
            },
        )

    image_path = find_image_path(object_id)

    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "image_not_found",
                "message": f"Image for object '{object_id}' not found",
            },
        )

    # Determine media type from extension
    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    # Fallback for unknown extensions is intentional
    media_type = media_types.get(ext)  # nosemgrep: no-dict-get-with-default
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(
        path=image_path,
        media_type=media_type,
        filename=f"{object_id}{ext}",
    )
