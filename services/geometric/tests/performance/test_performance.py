"""
Performance tests for the geometric service.

Tests measure latency and throughput under various conditions:
- Different image dimensions (ORB extraction scaling)
- Different max_features settings
- Match scenarios (extract only vs full match)
- Sequential and concurrent request patterns
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from typing import TYPE_CHECKING

import pytest

from .conftest import (
    CONCURRENCY_LEVELS,  # noqa: F401
    DIMENSION_SIZES,
    FEATURE_COUNT_LEVELS,
    ITERATIONS_PER_SCENARIO,
    THROUGHPUT_REQUESTS,  # noqa: F401
)
from .metrics import LatencyMetrics, PerformanceReport, ThroughputMetrics  # noqa: F401

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.mark.slow
@pytest.mark.performance
class TestDimensionLatency:
    """Test latency across different image dimensions."""

    @pytest.mark.parametrize(
        "size",
        DIMENSION_SIZES,
        ids=[f"{s}x{s}" for s in DIMENSION_SIZES],
    )
    def test_dimension_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """
        Measure extraction latency for different image dimensions.

        Args:
            client: Test client with real application
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            size: Image dimension (width and height)
        """
        image_key = f"dim_{size}"
        image_base64, image_size_kb = pregenerated_images[image_key]

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={"image": image_base64, "image_id": f"{image_key}_iter{i}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        # Record to report
        performance_report.add_dimension_result(size, metrics, image_size_kb)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Dimension Test: {size}x{size} ({image_size_kb:.0f} KB)")
        print(f"{'=' * 60}")
        print(metrics.summary())


@pytest.mark.slow
@pytest.mark.performance
class TestFeatureCountLatency:
    """Test latency across different max_features settings."""

    @pytest.mark.parametrize(
        "max_features",
        FEATURE_COUNT_LEVELS,
        ids=[f"{f}_features" for f in FEATURE_COUNT_LEVELS],
    )
    def test_feature_count_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        max_features: int,
    ) -> None:
        """
        Measure extraction latency for different max_features values.

        Args:
            client: Test client with real application
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            max_features: Maximum features to extract
        """
        image_base64, _ = pregenerated_images["feature_count"]

        metrics = LatencyMetrics()
        actual_features_sum = 0

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={
                    "image": image_base64,
                    "image_id": f"features_{max_features}_iter{i}",
                    "max_features": max_features,
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)
            actual_features_sum += response.json()["num_features"]

        avg_features = actual_features_sum / ITERATIONS_PER_SCENARIO

        # Record to report
        performance_report.add_feature_count_result(max_features, metrics, avg_features)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Feature Count Test: max={max_features}, avg_actual={avg_features:.0f}")
        print(f"{'=' * 60}")
        print(metrics.summary())


@pytest.mark.slow
@pytest.mark.performance
class TestMatchLatency:
    """Test latency for different matching scenarios."""

    def test_extract_only_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure baseline extraction latency (no matching).

        Establishes baseline for comparison with full match.
        """
        image_base64, _ = pregenerated_images["match_query"]
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/extract",
                json={"image": image_base64, "image_id": f"extract_only_{i}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_match_result("extract_only", metrics)

        print(f"\n{'=' * 60}")
        print("Match Test: extract_only (baseline)")
        print(f"{'=' * 60}")
        print(metrics.summary())

    def test_full_match_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure full match latency (extract + match + RANSAC).

        Tests the complete geometric verification pipeline.
        """
        query_b64, _ = pregenerated_images["match_query"]
        ref_b64, _ = pregenerated_images["match_reference"]

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/match",
                json={
                    "query_image": query_b64,
                    "reference_image": ref_b64,
                    "query_id": f"match_{i}",
                    "reference_id": "ref",
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_match_result("full_match", metrics)

        print(f"\n{'=' * 60}")
        print("Match Test: full_match (extract + match + RANSAC)")
        print(f"{'=' * 60}")
        print(metrics.summary())
