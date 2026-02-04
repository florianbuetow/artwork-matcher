"""
Shared fixtures for performance tests.

Performance tests use the real application with actual ORB extraction
and RANSAC verification. Images are pre-generated to avoid contaminating
latency measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from geometric_service.app import create_app
from geometric_service.config import clear_settings_cache
from geometric_service.core.state import reset_app_state

from .generators import (
    create_checkerboard_base64,
    create_noise_image_base64,
    create_transformed_image_base64,
    get_image_size_kb,
)
from .metrics import PerformanceReport

if TYPE_CHECKING:
    from collections.abc import Iterator


# Service root directory (services/geometric/)
SERVICE_ROOT = Path(__file__).parent.parent.parent

# Test configuration constants
ITERATIONS_PER_SCENARIO = 30
DIMENSION_SIZES = [100, 250, 500, 750, 1000, 1500, 2000]
FEATURE_COUNT_LEVELS = [100, 250, 500, 1000, 2000]
THROUGHPUT_REQUESTS = 250
CONCURRENCY_LEVELS = [2, 4, 8, 16]

# Report output path
REPORTS_DIR = SERVICE_ROOT / "reports" / "performance"
REPORT_PATH = REPORTS_DIR / "geometric_service_performance.md"


@pytest.fixture(scope="session")
def performance_report() -> Iterator[PerformanceReport]:
    """
    Session-scoped fixture that collects test results and writes report.

    The report is written to reports/performance/geometric_service_performance.md
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

    Uses context manager to trigger lifespan events.
    This client is shared across all tests in the module.
    """
    clear_settings_cache()
    reset_app_state()
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
def pregenerated_images() -> dict[str, tuple[str, float]]:
    """
    Pre-generate all test images before any tests run.

    This ensures image generation time is NOT included in latency
    measurements. All tests use identical pre-generated images.

    Returns:
        Dictionary mapping image key to (base64 string, size in KB) tuple.
    """
    images: dict[str, tuple[str, float]] = {}

    print("\n" + "=" * 60)
    print("Pre-generating test images...")
    print("=" * 60)

    # Dimension test images (noise for feature-rich content)
    print("\nDimension images:")
    for size in DIMENSION_SIZES:
        key = f"dim_{size}"
        image_b64 = create_noise_image_base64(size, size)
        image_size_kb = get_image_size_kb(image_b64)
        images[key] = (image_b64, image_size_kb)
        print(f"  {key}: {size}x{size} ({image_size_kb:.1f} KB)")

    # Feature count test image (large, consistent size)
    print("\nFeature count image:")
    feature_b64 = create_noise_image_base64(800, 800)
    feature_size = get_image_size_kb(feature_b64)
    images["feature_count"] = (feature_b64, feature_size)
    print(f"  feature_count: 800x800 ({feature_size:.1f} KB)")

    # Match test images (checkerboard for reliable matching)
    print("\nMatch test images:")
    query_b64 = create_checkerboard_base64(500, 500, block_size=25)
    query_size = get_image_size_kb(query_b64)
    images["match_query"] = (query_b64, query_size)
    print(f"  match_query: 500x500 checkerboard ({query_size:.1f} KB)")

    # Create transformed version for matching tests
    ref_b64 = create_transformed_image_base64(query_b64, rotation_deg=5, scale=0.95)
    ref_size = get_image_size_kb(ref_b64)
    images["match_reference"] = (ref_b64, ref_size)
    print(f"  match_reference: transformed ({ref_size:.1f} KB)")

    # Throughput test image (moderate size)
    print("\nThroughput image:")
    throughput_b64 = create_checkerboard_base64(400, 400, block_size=20)
    throughput_size = get_image_size_kb(throughput_b64)
    images["throughput"] = (throughput_b64, throughput_size)
    print(f"  throughput: 400x400 ({throughput_size:.1f} KB)")

    print(f"\nTotal pre-generated images: {len(images)}")
    print("=" * 60)
    return images
