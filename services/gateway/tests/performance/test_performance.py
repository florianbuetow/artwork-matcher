"""
Performance tests for the gateway service.

Tests measure latency and throughput for the gateway's endpoints and pipeline.
Backend services are mocked to measure gateway overhead only.

Test categories:
- Image dimension tests: varying input image sizes
- Endpoint latency tests: all gateway endpoints
- Geometric service check: availability status
- Throughput tests: sequential and concurrent requests
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pytest

from .conftest import (
    CONCURRENCY_LEVELS,
    DIMENSION_SIZES,
    ITERATIONS_PER_SCENARIO,
    THROUGHPUT_REQUESTS,
)
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
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """
        Measure identify latency for different image dimensions.

        The gateway processes images as base64 strings without decoding,
        so this tests JSON serialization overhead with varying payload sizes.

        Args:
            client: Test client with mocked backends
            pregenerated_images: Pre-generated test images (base64, size_kb tuples)
            performance_report: Report collector
            size: Image dimension (width and height)
        """
        image_key = f"dim_{size}"
        image_base64, image_size_kb = pregenerated_images[image_key]

        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/identify",
                json={"image": image_base64},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request {i} failed: {response.text}"
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
class TestEndpointLatency:
    """Test latency for each gateway endpoint."""

    def test_health_endpoint(
        self,
        client: TestClient,
        performance_report: PerformanceReport,
    ) -> None:
        """Measure health check endpoint latency."""
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.get("/health")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_endpoint_result("health", metrics)

        print(f"\n{'=' * 60}")
        print("Endpoint Test: /health")
        print(f"{'=' * 60}")
        print(metrics.summary())

    def test_info_endpoint(
        self,
        client: TestClient,
        performance_report: PerformanceReport,
    ) -> None:
        """Measure info endpoint latency."""
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.get("/info")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_endpoint_result("info", metrics)

        print(f"\n{'=' * 60}")
        print("Endpoint Test: /info")
        print(f"{'=' * 60}")
        print(metrics.summary())

    def test_identify_endpoint(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """Measure identify endpoint latency with standard image."""
        image_base64, _ = pregenerated_images["throughput"]
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/identify",
                json={"image": image_base64},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_endpoint_result("identify", metrics)

        print(f"\n{'=' * 60}")
        print("Endpoint Test: /identify")
        print(f"{'=' * 60}")
        print(metrics.summary())

    def test_objects_list_endpoint(
        self,
        client: TestClient,
        performance_report: PerformanceReport,
    ) -> None:
        """Measure objects list endpoint latency."""
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.get("/objects")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_endpoint_result("objects", metrics)

        print(f"\n{'=' * 60}")
        print("Endpoint Test: /objects")
        print(f"{'=' * 60}")
        print(metrics.summary())

    def test_objects_by_id_endpoint(
        self,
        client: TestClient,
        performance_report: PerformanceReport,
    ) -> None:
        """Measure objects by ID endpoint latency."""
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            # Use a known test object ID
            response = client.get("/objects/artwork_001")
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Accept 200 (found) or 404 (not found in test data)
            assert response.status_code in [200, 404], f"Request {i} failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_endpoint_result("objects_by_id", metrics)

        print(f"\n{'=' * 60}")
        print("Endpoint Test: /objects/{id}")
        print(f"{'=' * 60}")
        print(metrics.summary())


@pytest.mark.slow
@pytest.mark.performance
class TestGeometricService:
    """Check geometric service availability."""

    def test_geometric_service_status(
        self,
        geometric_service_status: tuple[bool, str],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Check if geometric service is available and record status.

        This test always passes but records whether the geometric service
        could be tested. The status is included in the performance report.

        The geometric_service_status fixture checks the REAL service
        before any mocks are active.
        """
        is_available, message = geometric_service_status

        if is_available:
            performance_report.set_geometric_service_status(
                available=True,
                message="Geometric service is available",
            )
            print(f"\n{'=' * 60}")
            print("Geometric Service: AVAILABLE")
            print(f"{'=' * 60}")
        else:
            performance_report.set_geometric_service_status(
                available=False,
                message=f"Could not test with geometric service: {message}",
            )
            print(f"\n{'=' * 60}")
            print(f"Geometric Service: NOT AVAILABLE - {message}")
            print(f"{'=' * 60}")


@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput (sequential and concurrent)."""

    def test_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
    ) -> None:
        """
        Measure throughput for sequential requests.

        Sends requests one after another and measures total time.
        This establishes the baseline for single-threaded performance.
        """
        image_base64, _ = pregenerated_images["throughput"]
        latency_metrics = LatencyMetrics()

        start_total = time.perf_counter()
        for i in range(THROUGHPUT_REQUESTS):
            start = time.perf_counter()
            response = client.post(
                "/identify",
                json={"image": image_base64},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request {i} failed: {response.text}"
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
        pregenerated_images: dict[str, tuple[str, float]],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """
        Measure throughput for concurrent requests.

        Sends requests in parallel using a thread pool to measure
        how well the gateway handles concurrent load.

        Args:
            client: Test client with mocked backends
            pregenerated_images: Pre-generated test images
            performance_report: Report collector
            workers: Number of concurrent workers
        """
        image_base64, _ = pregenerated_images["throughput"]
        requests_per_worker = THROUGHPUT_REQUESTS // workers

        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single request and return latency in ms."""
            start = time.perf_counter()
            response = client.post(
                "/identify",
                json={"image": image_base64},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                errors.append(f"Request {request_id} failed: {response.text}")

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
            total_requests=total_requests,
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
