"""
Client for the Geometric service.

Performs ORB feature extraction and RANSAC geometric verification.
"""

from __future__ import annotations

from typing import Any

from gateway.clients.base import BackendClient


class GeometricResult:
    """A single geometric verification result."""

    def __init__(self, data: dict[str, Any]) -> None:
        # External API response - required fields
        reference_id = data.get("reference_id")  # nosemgrep: no-dict-get-with-default
        if reference_id is None:
            msg = "reference_id missing in geometric result"
            raise ValueError(msg)
        self.reference_id: str = str(reference_id)

        is_match = data.get("is_match")  # nosemgrep: no-dict-get-with-default
        if is_match is None:
            msg = "is_match missing in geometric result"
            raise ValueError(msg)
        self.is_match: bool = bool(is_match)

        confidence = data.get("confidence")  # nosemgrep: no-dict-get-with-default
        if confidence is None:
            msg = "confidence missing in geometric result"
            raise ValueError(msg)
        self.confidence: float = float(confidence)

        # Optional fields
        inliers = data.get("inliers")  # nosemgrep: no-dict-get-with-default
        self.inliers: int = int(inliers) if inliers is not None else 0

        inlier_ratio = data.get("inlier_ratio")  # nosemgrep: no-dict-get-with-default
        self.inlier_ratio: float = float(inlier_ratio) if inlier_ratio is not None else 0.0


class BatchMatchResult:
    """Result from batch geometric matching."""

    def __init__(self, data: dict[str, Any]) -> None:
        # External API response fields
        self.query_id: str | None = data.get("query_id")  # nosemgrep: no-dict-get-with-default

        query_features = data.get("query_features")  # nosemgrep: no-dict-get-with-default
        self.query_features: int = int(query_features) if query_features is not None else 0

        results = data.get("results")  # nosemgrep: no-dict-get-with-default
        self.results: list[GeometricResult] = (
            [GeometricResult(r) for r in results] if results is not None else []
        )

        best_match = data.get("best_match")  # nosemgrep: no-dict-get-with-default
        self.best_match: GeometricResult | None = (
            GeometricResult(best_match) if best_match else None
        )

        processing_time_ms = data.get("processing_time_ms")  # nosemgrep: no-dict-get-with-default
        self.processing_time_ms: float = (
            float(processing_time_ms) if processing_time_ms is not None else 0.0
        )


class GeometricClient(BackendClient):
    """
    Client for Geometric service.

    Handles geometric verification via the Geometric service API.
    """

    # nosemgrep: no-default-parameter-values (optional tracing parameters)
    async def match(
        self,
        query_image: str,
        reference_image: str,
        query_id: str | None = None,
        reference_id: str | None = None,
    ) -> GeometricResult:
        """
        Verify geometric consistency between two images.

        Args:
            query_image: Base64-encoded query image
            reference_image: Base64-encoded reference image
            query_id: Optional query identifier
            reference_id: Optional reference identifier

        Returns:
            Geometric verification result
        """
        payload: dict[str, Any] = {
            "query_image": query_image,
            "reference_image": reference_image,
        }
        if query_id is not None:
            payload["query_id"] = query_id
        if reference_id is not None:
            payload["reference_id"] = reference_id

        result = await self._request("POST", "/match", json=payload)
        return GeometricResult(result)

    # nosemgrep: no-default-parameter-values (optional tracing parameter)
    async def match_batch(
        self,
        query_image: str,
        references: list[dict[str, str]],
        query_id: str | None = None,
    ) -> BatchMatchResult:
        """
        Verify query image against multiple reference images.

        Args:
            query_image: Base64-encoded query image
            references: List of references, each with 'reference_id' and 'reference_image'
            query_id: Optional query identifier

        Returns:
            Batch match result with all comparisons
        """
        payload: dict[str, Any] = {
            "query_image": query_image,
            "references": references,
        }
        if query_id is not None:
            payload["query_id"] = query_id

        result = await self._request("POST", "/match/batch", json=payload)
        return BatchMatchResult(result)
