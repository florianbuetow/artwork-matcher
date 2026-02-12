"""Unit tests for evaluation gateway client helpers."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import httpx

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluation.client import identify_image  # noqa: E402


class EvaluationClientTest(unittest.TestCase):
    """Validate response mapping and error handling in evaluation client."""

    def test_identify_image_maps_success_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "p1.jpg"
            image_path.write_bytes(b"fake-image-bytes")

            client = MagicMock(spec=httpx.Client)
            response = MagicMock(spec=httpx.Response)
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "match": {
                    "object_id": "o1",
                    "similarity_score": 0.91,
                    "geometric_score": 0.8,
                    "confidence": 0.86,
                },
                "alternatives": [
                    {
                        "object_id": "o2",
                        "similarity_score": 0.84,
                        "geometric_score": 0.6,
                        "confidence": 0.74,
                    }
                ],
                "timing": {
                    "embedding_ms": 12.5,
                    "search_ms": 9.0,
                    "geometric_ms": 15.0,
                    "total_ms": 36.5,
                },
            }
            client.post.return_value = response

            result = identify_image(
                client=client,
                gateway_url="http://localhost:8000",
                image_path=image_path,
                geometric_verification=True,
                k=10,
                threshold=0.0,
            )

            self.assertEqual(result.picture_id, "p1")
            self.assertEqual(result.mode, "geometric")
            self.assertEqual(result.matched_object_id, "o1")
            self.assertEqual(len(result.ranked_results), 2)
            self.assertEqual(result.ranked_results[0].object_id, "o1")
            self.assertAlmostEqual(result.total_ms, 36.5, places=6)
            self.assertIsNone(result.error)

            expected_image = base64.b64encode(b"fake-image-bytes").decode("ascii")
            payload = client.post.call_args.kwargs["json"]
            self.assertEqual(payload["image"], expected_image)

    def test_identify_image_maps_http_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "p2.jpg"
            image_path.write_bytes(b"fake-image-bytes")

            client = MagicMock(spec=httpx.Client)
            response = MagicMock(spec=httpx.Response)
            response.status_code = 502
            response.json.return_value = {
                "error": "backend_error",
                "message": "Backend unavailable",
            }

            http_error = httpx.HTTPStatusError(
                "Bad Gateway",
                request=MagicMock(),
                response=response,
            )
            client.post.return_value = response
            response.raise_for_status.side_effect = http_error

            result = identify_image(
                client=client,
                gateway_url="http://localhost:8000",
                image_path=image_path,
                geometric_verification=False,
                k=5,
                threshold=0.5,
            )

            self.assertEqual(result.mode, "embedding_only")
            self.assertEqual(result.error, "backend_error")
            self.assertIn("Backend unavailable", result.error_message or "")
            self.assertEqual(result.ranked_results, [])
            self.assertEqual(result.total_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
