"""
API routers for the search service.

Each router handles a specific domain of endpoints.
"""

from __future__ import annotations

from . import health, index, info, search

__all__ = ["health", "index", "info", "search"]
