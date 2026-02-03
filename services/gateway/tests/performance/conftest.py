"""
Shared fixtures for performance tests.

Performance tests use the real gateway application with HTTP-level mocking
for backend services. Test images are pre-generated to avoid contaminating
latency measurements.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import clear_settings_cache
from tests.factories import create_mock_embedding, create_test_image_base64

from .generators import create_noise_image_base64, get_image_size_kb
from .metrics import PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Iterator


# Service root directory (services/gateway/)
SERVICE_ROOT = Path(__file__).parent.parent.parent

# Test configuration constants
ITERATIONS_PER_SCENARIO = 30
THROUGHPUT_REQUESTS = 250
CONCURRENCY_LEVELS = [2, 4, 8, 16]

# Image dimension variations for latency tests (expanded like embeddings service)
DIMENSION_SIZES = [100, 250, 500, 750, 1000, 1500, 2000]

# Report output path (in service's reports/performance directory)
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "gateway_service_performance.md"

# Backend service URLs (must match config.yaml)
EMBEDDINGS_URL = "http://localhost:8001"
SEARCH_URL = "http://localhost:8002"
GEOMETRIC_URL = "http://localhost:8003"


def _check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open using socket connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _check_geometric_service_real() -> tuple[bool, str]:
    """
    Check if the geometric service is actually available (bypasses mocks).

    Uses socket to check port, then httpx for health check.
    This is called BEFORE any mocks are set up.
    """
    # First check if port is even open
    if not _check_port_open("localhost", 8003):
        return False, "Service not running"

    # Port is open, try HTTP health check
    return _check_geometric_http_endpoints()


def _check_geometric_http_endpoints() -> tuple[bool, str]:
    """Check geometric service HTTP endpoints."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{GEOMETRIC_URL}/health")
            if response.status_code != 200:
                return False, f"Health check returned {response.status_code}"

            info_response = client.get(f"{GEOMETRIC_URL}/info")
            if info_response.status_code != 200:
                return False, "Service /info endpoint not implemented"

            return True, "Service is available"
    except (httpx.ConnectError, httpx.TimeoutException):
        return False, "Service not running"
    except Exception as e:
        return False, str(e)


@pytest.fixture(scope="session")
def geometric_service_status() -> tuple[bool, str]:
    """
    Session-scoped fixture that checks geometric service status BEFORE mocks.

    This must run before any module-scoped mocks are set up to get the
    real service status.
    """
    return _check_geometric_service_real()


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written to reports/performance/gateway_service_performance.md
    when all tests complete.
    """
    report = PerformanceReport()
    yield report

    # Write report after all tests complete
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.generate_markdown())
    print(f"\n\nPerformance report written to: {REPORT_PATH}")


@pytest.fixture(scope="module")
def mock_backend_responses() -> Iterator[respx.MockRouter]:
    """
    Module-scoped mock for backend HTTP responses.

    Sets up fast mock responses for all backend services to measure
    gateway overhead without backend latency.
    """
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        # Mock embeddings service
        router.get(f"{EMBEDDINGS_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{EMBEDDINGS_URL}/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "service": "embeddings",
                    "version": "0.1.0",
                    "status": "healthy",
                    "model": {
                        "name": "facebook/dinov2-base",
                        "embedding_dimension": 768,
                    },
                },
            )
        )
        router.post(f"{EMBEDDINGS_URL}/embed").mock(
            return_value=httpx.Response(
                200,
                json={
                    "embedding": create_mock_embedding(768),
                    "dimension": 768,
                    "processing_time_ms": 1.0,
                    "image_id": None,
                },
            )
        )

        # Mock search service
        router.get(f"{SEARCH_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{SEARCH_URL}/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "service": "search",
                    "version": "0.1.0",
                    "status": "healthy",
                    "index": {"count": 100},
                },
            )
        )
        router.post(f"{SEARCH_URL}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "object_id": "artwork_001",
                            "score": 0.92,
                            "rank": 1,
                            "metadata": {"name": "Test Artwork 1"},
                        },
                        {
                            "object_id": "artwork_002",
                            "score": 0.85,
                            "rank": 2,
                            "metadata": {"name": "Test Artwork 2"},
                        },
                        {
                            "object_id": "artwork_003",
                            "score": 0.78,
                            "rank": 3,
                            "metadata": {"name": "Test Artwork 3"},
                        },
                    ],
                    "query_id": "test",
                    "k": 5,
                    "search_time_ms": 1.0,
                },
            )
        )

        # Mock geometric service
        router.get(f"{GEOMETRIC_URL}/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )
        router.get(f"{GEOMETRIC_URL}/info").mock(
            return_value=httpx.Response(
                200,
                json={
                    "service": "geometric",
                    "version": "0.1.0",
                    "status": "healthy",
                },
            )
        )
        router.post(f"{GEOMETRIC_URL}/match").mock(
            return_value=httpx.Response(
                200,
                json={
                    "match": True,
                    "inliers": 50,
                    "score": 0.85,
                    "processing_time_ms": 1.0,
                },
            )
        )
        router.post(f"{GEOMETRIC_URL}/match_batch").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"object_id": "artwork_001", "match": True, "inliers": 50, "score": 0.85},
                    ],
                    "processing_time_ms": 1.0,
                },
            )
        )

        yield router


@pytest.fixture(scope="module")
def performance_client(
    mock_backend_responses: respx.MockRouter,  # noqa: ARG001
) -> Iterator[TestClient]:
    """
    Create test client with the real gateway application.

    Uses context manager to trigger lifespan events.
    This client is shared across all tests in the module.
    """
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    clear_settings_cache()


@pytest.fixture
def client(performance_client: TestClient) -> TestClient:
    """Alias for performance_client for test compatibility."""
    return performance_client


@pytest.fixture(scope="module")
def pregenerated_images() -> dict[str, tuple[str, float]]:
    """
    Pre-generate all test images before any tests run.

    This ensures image generation time is NOT included in latency
    measurements. All tests use identical pre-generated images.

    Returns:
        Dictionary mapping image key to (base64 string, size in KB) tuple.
        Keys are formatted as:
        - "dim_{size}" for dimension variation tests
        - "throughput" for throughput tests
    """
    images: dict[str, tuple[str, float]] = {}

    print("\n" + "=" * 60)
    print("Pre-generating test images...")
    print("=" * 60)

    # Dimension test images (using noise for realistic testing)
    print("\nDimension images:")
    for size in DIMENSION_SIZES:
        key = f"dim_{size}"
        image_b64 = create_noise_image_base64(size, size)
        image_size_kb = get_image_size_kb(image_b64)
        images[key] = (image_b64, image_size_kb)
        print(f"  {key}: {size}x{size} ({image_size_kb:.1f} KB)")

    # Legacy keys for backwards compatibility with existing tests
    if "dim_100" in images:
        images["small"] = images["dim_100"]
    if "dim_500" in images:
        images["medium"] = images["dim_500"]
    if "dim_1000" in images:
        images["large"] = images["dim_1000"]

    # Throughput test image (standard size, smaller for faster tests)
    print("\nThroughput image:")
    throughput_b64 = create_test_image_base64(200, 200, "blue", "JPEG")
    throughput_size = get_image_size_kb(throughput_b64)
    images["throughput"] = (throughput_b64, throughput_size)
    print(f"  throughput: 200x200 ({throughput_size:.1f} KB)")

    print(f"\nTotal pre-generated images: {len(images)}")
    print("=" * 60)
    return images
