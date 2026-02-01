"""
Shared fixtures for performance tests.

Performance tests use the real application with the actual model loaded.
Images are pre-generated to avoid contaminating latency measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from embeddings_service.app import create_app
from embeddings_service.config import clear_settings_cache
from embeddings_service.core.state import reset_app_state

from .generators import (
    create_noise_image_base64,
    create_target_size_image_base64,
    get_image_size_kb,
)
from .metrics import PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Iterator


# Service root directory (services/embeddings/)
SERVICE_ROOT = Path(__file__).parent.parent.parent

# Test configuration constants
ITERATIONS_PER_SCENARIO = 30
DIMENSION_SIZES = [100, 512, 1024, 2048, 4096]
FILE_SIZE_TARGETS_KB = [10, 50, 100, 500, 1000, 2000, 5000]
THROUGHPUT_REQUESTS = 250
CONCURRENCY_LEVELS = [2, 4, 8, 16]

# Report output path (relative to service root)
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "embedding_service_performance.md"


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written to reports/embedding_service_performance.md
    when all tests complete.
    """
    report = PerformanceReport()
    yield report

    # Write report after all tests complete
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.generate_markdown())
    print(f"\n\nPerformance report written to: {REPORT_PATH}")


@pytest.fixture(scope="module")
def performance_client() -> Iterator[TestClient]:
    """
    Create test client with the real application.

    Uses context manager to trigger lifespan events (model loading).
    This client is shared across all tests in the module to avoid
    loading the model multiple times.
    """
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    # Clean up after all tests in this module
    clear_settings_cache()
    reset_app_state()


@pytest.fixture
def client(performance_client: TestClient) -> TestClient:
    """Alias for performance_client for test compatibility."""
    return performance_client


@pytest.fixture(scope="module")
def pregenerated_images() -> dict[str, str]:
    """
    Pre-generate all test images before any tests run.

    This ensures image generation time is NOT included in latency
    measurements. All tests use identical pre-generated images.

    Returns:
        Dictionary mapping image key to base64 string.
        Keys are formatted as:
        - "dim_{width}x{height}" for dimension tests
        - "size_{target_kb}kb" for file size tests
        - "throughput" for throughput tests
    """
    images: dict[str, str] = {}

    # Dimension test images (5 sizes)
    print("\nPre-generating dimension test images...")
    for size in DIMENSION_SIZES:
        key = f"dim_{size}x{size}"
        images[key] = create_noise_image_base64(size, size)
        actual_kb = get_image_size_kb(images[key])
        print(f"  {key}: {actual_kb:.1f} KB")

    # File size test images (5 target sizes)
    print("\nPre-generating file size test images...")
    for target_kb in FILE_SIZE_TARGETS_KB:
        key = f"size_{target_kb}kb"
        images[key] = create_target_size_image_base64(target_kb)
        actual_kb = get_image_size_kb(images[key])
        print(f"  {key}: target={target_kb} KB, actual={actual_kb:.1f} KB")

    # Throughput test image (single standard image at model input size)
    print("\nPre-generating throughput test image...")
    images["throughput"] = create_noise_image_base64(518, 518)
    actual_kb = get_image_size_kb(images["throughput"])
    print(f"  throughput: {actual_kb:.1f} KB")

    print(f"\nTotal pre-generated images: {len(images)}")
    return images
