"""Gateway client helpers for evaluation tooling."""

from __future__ import annotations

import base64
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import httpx

from evaluation.models import MatchResult, RankedResultItem

LOGGER = logging.getLogger(__name__)


def check_gateway_health(client: httpx.Client, gateway_url: str) -> dict[str, Any]:
    """Check gateway health status."""
    response = client.get(
        f"{gateway_url}/health", params={"check_backends": "true"}
    )
    response.raise_for_status()
    data = response.json()
    status = data.get("status")
    if status is None:
        msg = "Gateway health response missing 'status'"
        raise ValueError(msg)
    return data


def identify_image(
    client: httpx.Client,
    gateway_url: str,
    image_path: Path,
    geometric_verification: bool,
    k: int,
    threshold: float,
) -> MatchResult:
    """Call the /identify endpoint."""
    picture_id = image_path.stem
    mode = "geometric" if geometric_verification else "embedding_only"

    try:
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
    except OSError as e:
        LOGGER.error(
            "Failed to read evaluation image",
            extra={"picture_id": picture_id, "image_path": str(image_path)},
        )
        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            matched_object_id=None,
            similarity_score=None,
            geometric_score=None,
            confidence=None,
            ranked_results=[],
            embedding_ms=0.0,
            search_ms=0.0,
            geometric_ms=0.0,
            total_ms=0.0,
            error="file_error",
            error_message=str(e),
        )

    try:
        response = client.post(
            f"{gateway_url}/identify",
            json={
                "image": image_b64,
                "options": {
                    "k": k,
                    "threshold": threshold,
                    "geometric_verification": geometric_verification,
                    "include_alternatives": True,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        timing = data.get("timing", {})
        match = data.get("match")
        alternatives = data.get("alternatives", [])

        ranked: list[RankedResultItem] = []
        if match:
            parsed_match = _to_ranked_result_item(match, picture_id)
            if parsed_match is not None:
                ranked.append(parsed_match)
        for item in alternatives:
            parsed_alternative = _to_ranked_result_item(item, picture_id)
            if parsed_alternative is not None:
                ranked.append(parsed_alternative)

        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            matched_object_id=str(match.get("object_id")) if match else None,
            similarity_score=_to_optional_float(match.get("similarity_score")) if match else None,
            geometric_score=_to_optional_float(match.get("geometric_score")) if match else None,
            confidence=_to_optional_float(match.get("confidence")) if match else None,
            ranked_results=ranked,
            embedding_ms=_to_float(timing.get("embedding_ms"), default=0.0),
            search_ms=_to_float(timing.get("search_ms"), default=0.0),
            geometric_ms=_to_float(timing.get("geometric_ms"), default=0.0),
            total_ms=_to_float(timing.get("total_ms"), default=0.0),
            error=None,
            error_message=None,
        )

    except httpx.HTTPStatusError as e:
        LOGGER.warning(
            "Gateway /identify returned HTTP error",
            extra={
                "picture_id": picture_id,
                "status_code": e.response.status_code,
            },
        )
        error_data = {}
        try:
            error_data = e.response.json()
        except (JSONDecodeError, ValueError, TypeError):
            error_data = {}

        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            matched_object_id=None,
            similarity_score=None,
            geometric_score=None,
            confidence=None,
            ranked_results=[],
            embedding_ms=0.0,
            search_ms=0.0,
            geometric_ms=0.0,
            total_ms=0.0,
            error=error_data.get("error", "http_error"),
            error_message=error_data.get("message", str(e)),
        )

    except httpx.RequestError as e:
        LOGGER.error(
            "Gateway /identify request failed",
            extra={"picture_id": picture_id, "error": str(e)},
        )
        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            matched_object_id=None,
            similarity_score=None,
            geometric_score=None,
            confidence=None,
            ranked_results=[],
            embedding_ms=0.0,
            search_ms=0.0,
            geometric_ms=0.0,
            total_ms=0.0,
            error="request_error",
            error_message=str(e),
        )
    except (JSONDecodeError, ValueError, TypeError) as e:
        LOGGER.error(
            "Gateway /identify returned invalid response payload",
            extra={"picture_id": picture_id, "error": str(e)},
        )
        return MatchResult(
            picture_id=picture_id,
            mode=mode,
            matched_object_id=None,
            similarity_score=None,
            geometric_score=None,
            confidence=None,
            ranked_results=[],
            embedding_ms=0.0,
            search_ms=0.0,
            geometric_ms=0.0,
            total_ms=0.0,
            error="invalid_response",
            error_message=str(e),
        )


def _to_float(value: object, default: float) -> float:
    """Convert response value to float, using explicit fallback."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_float(value: object) -> float | None:
    """Convert response value to float, returning None on invalid input."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_ranked_result_item(item: dict[str, object], picture_id: str) -> RankedResultItem | None:
    """Convert one raw ranked result dictionary to a typed result item."""
    object_id = item.get("object_id")
    if object_id is None:
        LOGGER.warning(
            "Skipping ranked result without object_id",
            extra={"picture_id": picture_id},
        )
        return None

    return RankedResultItem(
        object_id=str(object_id),
        similarity_score=_to_optional_float(item.get("similarity_score")),
        geometric_score=_to_optional_float(item.get("geometric_score")),
        confidence=_to_optional_float(item.get("confidence")),
    )
