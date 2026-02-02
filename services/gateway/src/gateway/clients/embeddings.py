"""
Client for the Embeddings service.

Extracts DINOv2 embeddings from images.
"""

from __future__ import annotations

from typing import Any

from gateway.clients.base import BackendClient


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
        # External API response - handle missing field
        embedding = result.get("embedding")  # nosemgrep: no-dict-get-with-default
        if embedding is None or not isinstance(embedding, list):
            return []
        return [float(x) for x in embedding]

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
