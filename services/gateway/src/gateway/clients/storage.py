"""
Client for the Storage service.

Retrieves reference artwork images by object ID.
"""

from __future__ import annotations

import base64

import httpx

from gateway.clients.base import BackendClient
from gateway.core.exceptions import BackendError
from gateway.logging import get_logger


class StorageClient(BackendClient):
    """
    Client for Storage service.

    Handles retrieval of reference artwork images stored as raw bytes.
    All images are assumed to be JPEG format.
    """

    async def get_image_bytes(self, object_id: str) -> bytes | None:
        """
        Retrieve raw image bytes for an object.

        Args:
            object_id: Object identifier matching the FAISS index

        Returns:
            Raw image bytes, or None if object not found (404)

        Raises:
            BackendError: If storage service is unavailable or returns an error
        """
        logger = get_logger()
        path = f"/objects/{object_id}"

        await self._enforce_circuit_policy(path)

        try:
            response = await self.client.get(path)

            if response.status_code == 404:
                logger.debug(
                    "Object not found in storage",
                    extra={"object_id": object_id},
                )
                await self._record_success(path)
                return None

            response.raise_for_status()
            await self._record_success(path)
            return response.content

        except httpx.TimeoutException as e:
            await self._record_failure(path, "timeout")
            raise BackendError(
                error="backend_timeout",
                message="storage service timed out",
                status_code=504,
                details={
                    "backend": "storage",
                    "object_id": object_id,
                    "timeout_seconds": self.timeout,
                },
            ) from e

        except httpx.ConnectError as e:
            await self._record_failure(path, "connect_error")
            raise BackendError(
                error="backend_unavailable",
                message="storage service is not responding",
                status_code=502,
                details={
                    "backend": "storage",
                    "object_id": object_id,
                    "url": self.base_url,
                },
            ) from e

        except httpx.HTTPStatusError as e:
            await self._record_failure(path, "http_status_error")
            logger.warning(
                "Storage service error",
                extra={
                    "backend": "storage",
                    "object_id": object_id,
                    "status_code": e.response.status_code,
                },
            )
            raise BackendError(
                error="backend_error",
                message=f"storage service error: {e.response.status_code}",
                status_code=502,
                details={
                    "backend": "storage",
                    "object_id": object_id,
                    "backend_status_code": e.response.status_code,
                },
            ) from e

    async def get_image_base64(self, object_id: str) -> str | None:
        """
        Retrieve base64-encoded image for an object.

        Convenience wrapper around get_image_bytes for geometric verification.

        Args:
            object_id: Object identifier matching the FAISS index

        Returns:
            Base64-encoded image string, or None if object not found

        Raises:
            BackendError: If storage service is unavailable or returns an error
        """
        image_bytes = await self.get_image_bytes(object_id)
        if image_bytes is None:
            return None
        return base64.b64encode(image_bytes).decode("ascii")
