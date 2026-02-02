"""API routers for the gateway service."""

from gateway.routers import health, identify, info, objects

__all__ = ["health", "identify", "info", "objects"]
