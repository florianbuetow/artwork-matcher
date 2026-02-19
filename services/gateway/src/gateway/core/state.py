"""
Application state management.

Tracks runtime state like uptime, and stores HTTP client instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.clients import EmbeddingsClient, GeometricClient, SearchClient, StorageClient


@dataclass
class AppState:
    """
    Runtime application state.

    Attributes:
        start_time: When the application started (UTC)
        _embeddings_client: HTTP client for embeddings service (internal)
        _search_client: HTTP client for search service (internal)
        _geometric_client: HTTP client for geometric service (internal)
    """

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    _embeddings_client: EmbeddingsClient | None = field(default=None, repr=False)
    _search_client: SearchClient | None = field(default=None, repr=False)
    _geometric_client: GeometricClient | None = field(default=None, repr=False)
    _storage_client: StorageClient | None = field(default=None, repr=False)

    @property
    def embeddings_client(self) -> EmbeddingsClient:
        """Get the embeddings client. Raises RuntimeError if not initialized."""
        if self._embeddings_client is None:
            raise RuntimeError("Embeddings client not initialized")
        return self._embeddings_client

    @embeddings_client.setter
    def embeddings_client(self, value: EmbeddingsClient) -> None:
        """Set the embeddings client."""
        self._embeddings_client = value

    @property
    def search_client(self) -> SearchClient:
        """Get the search client. Raises RuntimeError if not initialized."""
        if self._search_client is None:
            raise RuntimeError("Search client not initialized")
        return self._search_client

    @search_client.setter
    def search_client(self, value: SearchClient) -> None:
        """Set the search client."""
        self._search_client = value

    @property
    def geometric_client(self) -> GeometricClient:
        """Get the geometric client. Raises RuntimeError if not initialized."""
        if self._geometric_client is None:
            raise RuntimeError("Geometric client not initialized")
        return self._geometric_client

    @geometric_client.setter
    def geometric_client(self, value: GeometricClient) -> None:
        """Set the geometric client."""
        self._geometric_client = value

    @property
    def storage_client(self) -> StorageClient:
        """Get the blob store client. Raises RuntimeError if not initialized."""
        if self._storage_client is None:
            raise RuntimeError("Blob store client not initialized")
        return self._storage_client

    @storage_client.setter
    def storage_client(self, value: StorageClient) -> None:
        """Set the blob store client."""
        self._storage_client = value

    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        now = datetime.now(UTC)
        delta = now - self.start_time
        return delta.total_seconds()

    @property
    def uptime_formatted(self) -> str:
        """
        Format uptime as human-readable string.

        Returns:
            String like "2d 3h 15m 42s" or "15m 42s"
        """
        seconds = int(self.uptime_seconds)
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)


# Global application state instance
# Initialized in lifespan context
_app_state: AppState | None = None


def get_app_state() -> AppState:
    """
    Get the current application state.

    Raises:
        RuntimeError: If called before app startup
    """
    if _app_state is None:
        raise RuntimeError("Application state not initialized")
    return _app_state


def init_app_state() -> AppState:
    """Initialize application state. Called during startup."""
    # nosemgrep: config.semgrep.python.no-noqa-for-typing
    global _app_state  # noqa: PLW0603 - intentional singleton pattern
    _app_state = AppState()
    return _app_state


def reset_app_state() -> None:
    """Reset application state. Used in testing."""
    # nosemgrep: config.semgrep.python.no-noqa-for-typing
    global _app_state  # noqa: PLW0603 - intentional singleton pattern
    _app_state = None
