"""
Client for the Search service.

Performs FAISS vector similarity search.
"""

from __future__ import annotations

from typing import Any

from gateway.clients.base import BackendClient


class SearchResult:
    """A single search result from the Search service."""

    def __init__(self, data: dict[str, Any]) -> None:
        # External API response - access fields, raise if missing required fields
        object_id = data.get("object_id")  # nosemgrep: no-dict-get-with-default
        if object_id is None:
            msg = "object_id missing in search result"
            raise ValueError(msg)
        self.object_id: str = str(object_id)

        score = data.get("score")  # nosemgrep: no-dict-get-with-default
        if score is None:
            msg = "score missing in search result"
            raise ValueError(msg)
        self.score: float = float(score)

        rank = data.get("rank")  # nosemgrep: no-dict-get-with-default
        self.rank: int = int(rank) if rank is not None else 0

        metadata = data.get("metadata")  # nosemgrep: no-dict-get-with-default
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}


class SearchClient(BackendClient):
    """
    Client for Search service.

    Handles vector similarity search via the Search service API.
    """

    async def search(
        self,
        embedding: list[float],
        k: int,
        threshold: float,
    ) -> list[SearchResult]:
        """
        Search for similar vectors in the index.

        Args:
            embedding: Query embedding vector
            k: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of search results, ordered by score descending
        """
        result = await self._request(
            "POST",
            "/search",
            json={
                "embedding": embedding,
                "k": k,
                "threshold": threshold,
            },
        )

        results = result.get("results")  # nosemgrep: no-dict-get-with-default
        if results is None:
            return []
        return [SearchResult(r) for r in results]

    async def get_index_count(self) -> int:
        """
        Get the number of items in the index.

        Returns:
            Index count

        Raises:
            KeyError: If index info not in service response
        """
        info = await self.get_info()
        index_info = info.get("index")  # nosemgrep: no-dict-get-with-default
        if index_info is None:
            msg = "index info not in search service response"
            raise KeyError(msg)
        count = index_info.get("count")  # nosemgrep: no-dict-get-with-default
        if count is None:
            msg = "count not in index info"
            raise KeyError(msg)
        return int(count)

    async def is_index_loaded(self) -> bool:
        """
        Check if the index is loaded.

        Returns:
            True if index is loaded and ready

        Raises:
            KeyError: If index info not in service response
        """
        info = await self.get_info()
        index_info = info.get("index")  # nosemgrep: no-dict-get-with-default
        if index_info is None:
            msg = "index info not in search service response"
            raise KeyError(msg)
        is_loaded = index_info.get("is_loaded")  # nosemgrep: no-dict-get-with-default
        if is_loaded is None:
            msg = "is_loaded not in index info"
            raise KeyError(msg)
        return bool(is_loaded)
