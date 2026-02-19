"""API routers for the storage service."""

from storage_service.routers import health, info, objects

__all__ = ["health", "info", "objects"]
