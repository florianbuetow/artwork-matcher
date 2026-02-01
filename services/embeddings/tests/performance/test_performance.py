"""
Performance tests for the embeddings service.

Tests measure latency and throughput under various conditions:
- Different image dimensions
- Different compressed file sizes
- Sequential and concurrent request patterns
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pytest

from .conftest import (
    CONCURRENCY_LEVELS,
    DIMENSION_SIZES,
    FILE_SIZE_TARGETS_KB,
    ITERATIONS_PER_SCENARIO,
    THROUGHPUT_REQUESTS,
)
from .generators import get_image_size_kb
from .metrics import LatencyMetrics, PerformanceReport, ThroughputMetrics

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
        pregenerated_images: dict[str, str],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """
        Measure embedding latency for different image dimensions.

        Args:
            client: Test client with real model loaded
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            size: Image dimension (width and height)
        """
        image_key = f"dim_{size}x{size}"
        image_base64 = pregenerated_images[image_key]
        image_size_kb = get_image_size_kb(image_base64)

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            image_id = f"{image_key}_iter{i}"
            start = time.perf_counter()
            response = client.post("/embed", json={"image": image_base64, "image_id": image_id})
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
class TestFileSizeLatency:
    """Test latency across different compressed file sizes."""

    @pytest.mark.parametrize(
        "target_kb",
        FILE_SIZE_TARGETS_KB,
        ids=[f"{kb}kb" for kb in FILE_SIZE_TARGETS_KB],
    )
    def test_filesize_latency(
        self,
        client: TestClient,
        pregenerated_images: dict[str, str],
        performance_report: PerformanceReport,
        target_kb: int,
    ) -> None:
        """
        Measure embedding latency for different file sizes.

        Args:
            client: Test client with real model loaded
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            target_kb: Target file size in kilobytes
        """
        image_key = f"size_{target_kb}kb"
        image_base64 = pregenerated_images[image_key]
        actual_size_kb = get_image_size_kb(image_base64)

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            image_id = f"{image_key}_iter{i}"
            start = time.perf_counter()
            response = client.post("/embed", json={"image": image_base64, "image_id": image_id})
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        # Record to report
        performance_report.add_filesize_result(target_kb, metrics, actual_size_kb)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"File Size Test: target={target_kb} KB, actual={actual_size_kb:.0f} KB")
        print(f"{'=' * 60}")
        print(metrics.summary())


@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput (sequential and concurrent)."""

    def test_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_images: dict[str, str],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure throughput for sequential requests.

        Sends requests one after another and measures total time.
        """
        image_base64 = pregenerated_images["throughput"]

        latency_metrics = LatencyMetrics()

        start_total = time.perf_counter()
        for i in range(THROUGHPUT_REQUESTS):
            image_id = f"throughput_seq_{i}"
            start = time.perf_counter()
            response = client.post("/embed", json={"image": image_base64, "image_id": image_id})
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            latency_metrics.add(elapsed_ms)

        total_duration = time.perf_counter() - start_total

        throughput = ThroughputMetrics(
            total_requests=THROUGHPUT_REQUESTS,
            total_duration_seconds=total_duration,
        )

        # Record to report
        performance_report.add_throughput_result("sequential", latency_metrics, throughput)

        # Print results
        print(f"\n{'=' * 60}")
        print("Sequential Throughput Test")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())

    @pytest.mark.parametrize(
        "workers",
        CONCURRENCY_LEVELS,
        ids=[f"{w}_workers" for w in CONCURRENCY_LEVELS],
    )
    def test_concurrent_throughput(
        self,
        client: TestClient,
        pregenerated_images: dict[str, str],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """
        Measure throughput for concurrent requests.

        Sends requests in parallel using a thread pool.

        Args:
            client: Test client with real model loaded
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            workers: Number of concurrent workers
        """
        image_base64 = pregenerated_images["throughput"]
        requests_per_worker = THROUGHPUT_REQUESTS // workers

        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single request and return latency in ms."""
            image_id = f"throughput_c{workers}_{request_id}"
            start = time.perf_counter()
            response = client.post("/embed", json={"image": image_base64, "image_id": image_id})
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                errors.append(f"Request failed: {response.text}")

            return elapsed_ms

        start_total = time.perf_counter()
        total_requests = requests_per_worker * workers

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(total_requests)]

            for future in as_completed(futures):
                latency_ms = future.result()
                latency_metrics.add(latency_ms)

        total_duration = time.perf_counter() - start_total

        # Check for errors
        assert not errors, f"Some requests failed: {errors[:5]}"

        throughput = ThroughputMetrics(
            total_requests=requests_per_worker * workers,
            total_duration_seconds=total_duration,
        )

        # Record to report
        performance_report.add_throughput_result(
            f"concurrent_{workers}", latency_metrics, throughput
        )

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Concurrent Throughput Test ({workers} workers)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())
