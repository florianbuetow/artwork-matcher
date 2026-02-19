"""
Performance tests for the storage service.

Tests measure latency and throughput under various conditions:
- Different object sizes for PUT/GET/DELETE
- Sequential request throughput
- Concurrent request throughput
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import pytest

from .conftest import (
    CONCURRENCY_LEVELS,
    ITERATIONS_PER_SCENARIO,
    OBJECT_SIZES,
    THROUGHPUT_OBJECT_SIZE,
    THROUGHPUT_REQUESTS,
)
from .metrics import LatencyMetrics, PerformanceReport, ThroughputMetrics

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def delete_all_objects(client: TestClient) -> None:
    """Delete all objects and verify successful cleanup."""
    response = client.delete("/objects")
    assert response.status_code == 200, f"Failed to clean up objects: {response.text}"


@pytest.mark.slow
@pytest.mark.performance
class TestObjectSizeLatency:
    """Test latency across different object sizes."""

    @pytest.mark.parametrize(
        "size",
        OBJECT_SIZES,
        ids=[f"{size}_bytes" for size in OBJECT_SIZES],
    )
    def test_put_latency(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """Measure PUT latency for different object sizes."""
        blob = pregenerated_data["blobs"][size]
        metrics = LatencyMetrics()

        for i in range(ITERATIONS_PER_SCENARIO):
            object_id = f"perf_put_{size}_{i}"
            start = time.perf_counter()
            response = client.put(f"/objects/{object_id}", content=blob)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 204, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_object_size_result("put", size, metrics)

        print(f"\n{'=' * 60}")
        print(f"PUT Object Size Test: {size:,} bytes")
        print(f"{'=' * 60}")
        print(metrics.summary())

        delete_all_objects(client)

    @pytest.mark.parametrize(
        "size",
        OBJECT_SIZES,
        ids=[f"{size}_bytes" for size in OBJECT_SIZES],
    )
    def test_get_latency(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """Measure GET latency for different object sizes."""
        blob = pregenerated_data["blobs"][size]
        object_ids = [f"perf_get_{size}_{i}" for i in range(ITERATIONS_PER_SCENARIO)]

        for object_id in object_ids:
            put_response = client.put(f"/objects/{object_id}", content=blob)
            assert put_response.status_code == 204, f"Failed to create object: {put_response.text}"

        metrics = LatencyMetrics()
        for object_id in object_ids:
            start = time.perf_counter()
            response = client.get(f"/objects/{object_id}")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_object_size_result("get", size, metrics)

        print(f"\n{'=' * 60}")
        print(f"GET Object Size Test: {size:,} bytes")
        print(f"{'=' * 60}")
        print(metrics.summary())

        delete_all_objects(client)

    @pytest.mark.parametrize(
        "size",
        OBJECT_SIZES,
        ids=[f"{size}_bytes" for size in OBJECT_SIZES],
    )
    def test_delete_latency(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """Measure DELETE latency for different object sizes."""
        blob = pregenerated_data["blobs"][size]
        object_ids = [f"perf_delete_{size}_{i}" for i in range(ITERATIONS_PER_SCENARIO)]

        for object_id in object_ids:
            put_response = client.put(f"/objects/{object_id}", content=blob)
            assert put_response.status_code == 204, f"Failed to create object: {put_response.text}"

        metrics = LatencyMetrics()
        for object_id in object_ids:
            start = time.perf_counter()
            response = client.delete(f"/objects/{object_id}")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 204, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_object_size_result("delete", size, metrics)

        print(f"\n{'=' * 60}")
        print(f"DELETE Object Size Test: {size:,} bytes")
        print(f"{'=' * 60}")
        print(metrics.summary())

        delete_all_objects(client)


@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput (sequential and concurrent)."""

    def test_put_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
    ) -> None:
        """Measure throughput for sequential PUT requests."""
        blob = pregenerated_data["throughput_blob"]
        latency_metrics = LatencyMetrics()

        start_total = time.perf_counter()
        for i in range(THROUGHPUT_REQUESTS):
            object_id = f"perf_put_sequential_{i}"
            start = time.perf_counter()
            response = client.put(f"/objects/{object_id}", content=blob)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 204, f"Request failed: {response.text}"
            latency_metrics.add(elapsed_ms)

        total_duration = time.perf_counter() - start_total
        throughput = ThroughputMetrics(
            total_requests=THROUGHPUT_REQUESTS,
            total_duration_seconds=total_duration,
        )
        performance_report.add_throughput_result("put_sequential", latency_metrics, throughput)

        print(f"\n{'=' * 60}")
        print(f"Sequential PUT Throughput Test ({THROUGHPUT_OBJECT_SIZE:,} bytes)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())

        delete_all_objects(client)

    def test_get_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
    ) -> None:
        """Measure throughput for sequential GET requests."""
        blob = pregenerated_data["throughput_blob"]
        target_id = "perf_get_sequential_target"

        put_response = client.put(f"/objects/{target_id}", content=blob)
        assert put_response.status_code == 204, f"Failed to create object: {put_response.text}"

        latency_metrics = LatencyMetrics()
        start_total = time.perf_counter()
        for _ in range(THROUGHPUT_REQUESTS):
            start = time.perf_counter()
            response = client.get(f"/objects/{target_id}")
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            latency_metrics.add(elapsed_ms)

        total_duration = time.perf_counter() - start_total
        throughput = ThroughputMetrics(
            total_requests=THROUGHPUT_REQUESTS,
            total_duration_seconds=total_duration,
        )
        performance_report.add_throughput_result("get_sequential", latency_metrics, throughput)

        print(f"\n{'=' * 60}")
        print(f"Sequential GET Throughput Test ({THROUGHPUT_OBJECT_SIZE:,} bytes)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())

        delete_all_objects(client)

    @pytest.mark.parametrize(
        "workers",
        CONCURRENCY_LEVELS,
        ids=[f"{workers}_workers" for workers in CONCURRENCY_LEVELS],
    )
    def test_put_concurrent_throughput(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """Measure throughput for concurrent PUT requests."""
        blob = pregenerated_data["throughput_blob"]
        requests_per_worker = THROUGHPUT_REQUESTS // workers
        total_requests = requests_per_worker * workers
        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single PUT request and return latency in ms."""
            object_id = f"perf_put_concurrent_{workers}_{request_id}"
            start = time.perf_counter()
            response = client.put(f"/objects/{object_id}", content=blob)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 204:
                errors.append(f"Request {request_id} failed: {response.text}")

            return elapsed_ms

        start_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(total_requests)]
            for future in as_completed(futures):
                latency_metrics.add(future.result())
        total_duration = time.perf_counter() - start_total

        assert not errors, f"Some requests failed: {errors[:5]}"

        throughput = ThroughputMetrics(
            total_requests=total_requests,
            total_duration_seconds=total_duration,
        )
        performance_report.add_throughput_result(
            f"put_concurrent_{workers}",
            latency_metrics,
            throughput,
        )

        print(f"\n{'=' * 60}")
        print(f"Concurrent PUT Throughput Test ({workers} workers)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())

        delete_all_objects(client)

    @pytest.mark.parametrize(
        "workers",
        CONCURRENCY_LEVELS,
        ids=[f"{workers}_workers" for workers in CONCURRENCY_LEVELS],
    )
    def test_get_concurrent_throughput(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """Measure throughput for concurrent GET requests."""
        blob = pregenerated_data["throughput_blob"]
        target_id = f"perf_get_concurrent_target_{workers}"
        requests_per_worker = THROUGHPUT_REQUESTS // workers
        total_requests = requests_per_worker * workers

        put_response = client.put(f"/objects/{target_id}", content=blob)
        assert put_response.status_code == 204, f"Failed to create object: {put_response.text}"

        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single GET request and return latency in ms."""
            start = time.perf_counter()
            response = client.get(f"/objects/{target_id}")
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                errors.append(f"Request {request_id} failed: {response.text}")

            return elapsed_ms

        start_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(total_requests)]
            for future in as_completed(futures):
                latency_metrics.add(future.result())
        total_duration = time.perf_counter() - start_total

        assert not errors, f"Some requests failed: {errors[:5]}"

        throughput = ThroughputMetrics(
            total_requests=total_requests,
            total_duration_seconds=total_duration,
        )
        performance_report.add_throughput_result(
            f"get_concurrent_{workers}",
            latency_metrics,
            throughput,
        )

        print(f"\n{'=' * 60}")
        print(f"Concurrent GET Throughput Test ({workers} workers)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())

        delete_all_objects(client)
