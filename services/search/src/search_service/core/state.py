"""
Application state management.

Tracks runtime state like uptime, and stores FAISS index reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from search_service.services.faiss_index import FAISSIndex


@dataclass
class AppState:
    """
    Runtime application state.

    Attributes:
        start_time: When the application started (UTC)
        faiss_index: Loaded FAISS index wrapper instance
        index_load_error: Error message if index auto-load failed at startup
    """

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    faiss_index: FAISSIndex | None = None
    index_load_error: str | None = None

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

    @property
    def index_loaded(self) -> bool:
        """Check if the FAISS index is loaded and ready."""
        return self.faiss_index is not None

    @property
    def index_count(self) -> int:
        """Get the number of vectors in the index."""
        if self.faiss_index is None:
            return 0
        return self.faiss_index.count


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
