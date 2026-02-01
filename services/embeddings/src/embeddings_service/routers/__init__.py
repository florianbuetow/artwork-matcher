"""
API routers for the embeddings service.

Each router handles a specific domain of endpoints.
"""

from embeddings_service.routers import embed, health, info

__all__ = ["embed", "health", "info"]
