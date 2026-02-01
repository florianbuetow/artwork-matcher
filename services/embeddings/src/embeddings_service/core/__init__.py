"""
Core components for the embeddings service.

Provides application lifecycle management, state tracking,
and exception handling.
"""

from embeddings_service.core.exceptions import ServiceError
from embeddings_service.core.state import AppState, get_app_state

__all__ = ["AppState", "ServiceError", "get_app_state"]
