"""
Shared fixtures for storage service performance tests.

Performance tests use the real application with a temporary storage directory.
Blobs are pre-generated to avoid contaminating latency measurements.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from storage_service.app import create_app
from storage_service.config import clear_settings_cache
from storage_service.core.state import reset_app_state

from .generators import create_random_blob, format_size
from .metrics import PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Iterator


# Service root directory (services/storage/)
SERVICE_ROOT = Path(__file__).parent.parent.parent

# Test configuration constants
ITERATIONS_PER_SCENARIO = 30
OBJECT_SIZES = [
    10_240,  # 10 KB - small thumbnail
    51_200,  # 50 KB - compressed web image
    102_400,  # 100 KB - typical phone photo (compressed)
    512_000,  # 500 KB - high-quality JPEG
    1_048_576,  # 1 MB - large photo
    2_097_152,  # 2 MB - high-res photo
    5_242_880,  # 5 MB - uncompressed / RAW-ish
]
THROUGHPUT_REQUESTS = 250
THROUGHPUT_OBJECT_SIZE = 512_000  # 500 KB for throughput tests (typical image)
CONCURRENCY_LEVELS = [2, 4, 8, 16]

# Report output path
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "storage_service_performance.md"


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written to reports/performance/storage_service_performance.md
    when all tests complete.
    """
    report = PerformanceReport()
    yield report

    # Write report after all tests complete
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.generate_markdown())
    print(f"\n\nPerformance report written to: {REPORT_PATH}")


@pytest.fixture(scope="module")
def performance_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """
    Create test client with the real application using a temp storage directory.

    Uses a temporary directory for storage to avoid polluting production data.
    Uses context manager to trigger lifespan events (blob store initialization).
    """
    tmp_storage = tmp_path_factory.mktemp("storage_perf")

    clear_settings_cache()
    reset_app_state()

    # Override storage path via environment variable
    os.environ["STORAGE__STORAGE__PATH"] = str(tmp_storage)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    os.environ.pop("STORAGE__STORAGE__PATH", None)
    clear_settings_cache()
    reset_app_state()


@pytest.fixture
def client(performance_client: TestClient) -> TestClient:
    """Alias for performance_client for test compatibility."""
    return performance_client


@pytest.fixture(scope="module")
def pregenerated_data() -> dict[str, Any]:
    """
    Pre-generate all test blobs before tests run.

    This ensures blob generation time is NOT included in latency
    measurements. All tests use identical pre-generated data.

    Returns:
        Dictionary with:
        - "blobs": Dict mapping size_bytes to random bytes
        - "throughput_blob": Single blob for throughput tests
    """
    blobs: dict[int, bytes] = {}

    print("\nPre-generating test blobs...")
    for size in OBJECT_SIZES:
        blobs[size] = create_random_blob(size)
        print(f"  {format_size(size)}: {len(blobs[size]):,} bytes")

    throughput_blob = create_random_blob(THROUGHPUT_OBJECT_SIZE)
    print(f"  Throughput blob: {format_size(THROUGHPUT_OBJECT_SIZE)}")

    print(f"\nTotal pre-generated: {len(blobs)} size variants + 1 throughput blob")

    return {
        "blobs": blobs,
        "throughput_blob": throughput_blob,
    }
