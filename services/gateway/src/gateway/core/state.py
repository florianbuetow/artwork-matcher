"""
Application state management.

Tracks runtime state like uptime, and stores HTTP client instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class AppState:
    """
    Runtime application state.

    Attributes:
        start_time: When the application started (UTC)
        embeddings_client: HTTP client for embeddings service
        search_client: HTTP client for search service
        geometric_client: HTTP client for geometric service
    """

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    embeddings_client: Any = None  # EmbeddingsClient
    search_client: Any = None  # SearchClient
    geometric_client: Any = None  # GeometricClient

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
