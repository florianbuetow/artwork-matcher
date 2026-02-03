"""
Client for the Embeddings service.

Extracts DINOv2 embeddings from images.
"""

from __future__ import annotations

from typing import Any

from gateway.clients.base import BackendClient
from gateway.core.exceptions import BackendError


class EmbeddingsClient(BackendClient):
    """
    Client for Embeddings service.

    Handles embedding extraction from images via the Embeddings service API.
    """

    # nosemgrep: no-default-parameter-values (optional tracing parameter)
    async def embed(self, image_b64: str, image_id: str | None = None) -> list[float]:
        """
        Extract embedding from image.

        Args:
            image_b64: Base64-encoded image data
            image_id: Optional identifier for logging/tracing

        Returns:
            Embedding vector as list of floats
        """
        payload: dict[str, Any] = {"image": image_b64}
        if image_id is not None:
            payload["image_id"] = image_id

        result = await self._request("POST", "/embed", json=payload)

        embedding = result.get("embedding")  # nosemgrep: no-dict-get-with-default
        if embedding is None:
            raise BackendError(
                error="invalid_response",
                message="Embeddings service returned response without 'embedding' field",
                status_code=502,
                details={"backend": "embeddings", "response_keys": list(result.keys())},
            )
        if not isinstance(embedding, list):
            embed_type = type(embedding).__name__
            raise BackendError(
                error="invalid_response",
                message=f"Embeddings service returned non-list embedding: {embed_type}",
                status_code=502,
                details={"backend": "embeddings", "embedding_type": embed_type},
            )
        if len(embedding) == 0:
            raise BackendError(
                error="empty_embedding",
                message="Embeddings service returned empty embedding vector",
                status_code=502,
                details={"backend": "embeddings"},
            )

        try:
            return [float(x) for x in embedding]
        except (ValueError, TypeError) as e:
            raise BackendError(
                error="invalid_response",
                message=f"Embeddings service returned non-numeric values in embedding: {e}",
                status_code=502,
                details={"backend": "embeddings"},
            ) from e

    async def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension from service info.

        Returns:
            Embedding dimension (e.g., 768)

        Raises:
            KeyError: If embedding dimension not in service info
        """
        info = await self.get_info()
        model_info = info.get("model")  # nosemgrep: no-dict-get-with-default
        if model_info is None:
            msg = "model info not in embeddings service response"
            raise KeyError(msg)
        dimension = model_info.get("embedding_dimension")  # nosemgrep: no-dict-get-with-default
        if dimension is None:
            msg = "embedding_dimension not in model info"
            raise KeyError(msg)
        return int(dimension)
