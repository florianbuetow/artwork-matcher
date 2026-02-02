"""
Performance tests for the search service.

Tests measure latency and throughput under various conditions:
- Different index sizes (number of vectors)
- Different k values (number of results)
- Sequential and concurrent request patterns
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import pytest

from .conftest import (
    CONCURRENCY_LEVELS,
    INDEX_SIZES,
    ITERATIONS_PER_SCENARIO,
    K_VALUES,
    THROUGHPUT_REQUESTS,
)
from .metrics import LatencyMetrics, PerformanceReport, ThroughputMetrics

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def populate_index(
    client: TestClient,
    embeddings: list[list[float]],
    count: int,
) -> None:
    """
    Populate the index with the specified number of embeddings.

    Clears existing index first, then adds vectors one by one.

    Args:
        client: Test client
        embeddings: Pre-generated embeddings to add
        count: Number of embeddings to add
    """
    # Clear existing index
    client.delete("/index")

    # Add embeddings
    for i in range(count):
        response = client.post(
            "/add",
            json={
                "object_id": f"obj_{i}",
                "embedding": embeddings[i],
            },
        )
        assert response.status_code == 201, f"Failed to add embedding {i}: {response.text}"


@pytest.mark.slow
@pytest.mark.performance
class TestIndexSizeLatency:
    """Test latency across different index sizes."""

    @pytest.mark.parametrize(
        "size",
        INDEX_SIZES,
        ids=[f"{s}_vectors" for s in INDEX_SIZES],
    )
    def test_index_size_latency(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        size: int,
    ) -> None:
        """
        Measure search latency for different index sizes.

        Args:
            client: Test client
            pregenerated_data: Pre-generated embeddings
            performance_report: Report collector
            size: Number of vectors to add to index
        """
        # Populate index (not timed)
        print(f"\nPopulating index with {size:,} vectors...")
        populate_index(client, pregenerated_data["embeddings"], size)

        query = pregenerated_data["query"]
        metrics = LatencyMetrics()

        # Use consistent k=5 for index size tests
        for _ in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/search",
                json={"embedding": query, "k": 5},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_index_size_result(size, metrics)

        print(f"\n{'=' * 60}")
        print(f"Index Size Test: {size:,} vectors")
        print(f"{'=' * 60}")
        print(metrics.summary())


@pytest.mark.slow
@pytest.mark.performance
class TestKValueLatency:
    """Test latency across different k values."""

    @pytest.fixture(autouse=True)
    def setup_index(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
    ) -> None:
        """Populate index with fixed size for k tests."""
        # Use 10,000 vectors as representative size
        print("\nPopulating index with 10,000 vectors for k-value tests...")
        populate_index(client, pregenerated_data["embeddings"], 10000)

    @pytest.mark.parametrize(
        "k",
        K_VALUES,
        ids=[f"k={k}" for k in K_VALUES],
    )
    def test_k_value_latency(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        k: int,
    ) -> None:
        """
        Measure search latency for different k values.

        Args:
            client: Test client
            pregenerated_data: Pre-generated embeddings
            performance_report: Report collector
            k: Number of results to request
        """
        query = pregenerated_data["query"]
        metrics = LatencyMetrics()

        for _ in range(ITERATIONS_PER_SCENARIO):
            start = time.perf_counter()
            response = client.post(
                "/search",
                json={"embedding": query, "k": k},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            metrics.add(elapsed_ms)

        performance_report.add_k_value_result(k, metrics)

        print(f"\n{'=' * 60}")
        print(f"K Value Test: k={k}")
        print(f"{'=' * 60}")
        print(metrics.summary())


@pytest.mark.slow
@pytest.mark.performance
class TestThroughput:
    """Test request throughput (sequential and concurrent)."""

    @pytest.fixture(autouse=True)
    def setup_index(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
    ) -> None:
        """Populate index for throughput tests."""
        # Use 10,000 vectors as representative size
        print("\nPopulating index with 10,000 vectors for throughput tests...")
        populate_index(client, pregenerated_data["embeddings"], 10000)

    def test_sequential_throughput(
        self,
        client: TestClient,
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
    ) -> None:
        """Measure throughput for sequential requests."""
        query = pregenerated_data["query"]
        latency_metrics = LatencyMetrics()

        start_total = time.perf_counter()
        for _ in range(THROUGHPUT_REQUESTS):
            start = time.perf_counter()
            response = client.post(
                "/search",
                json={"embedding": query, "k": 5},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200, f"Request failed: {response.text}"
            latency_metrics.add(elapsed_ms)

        total_duration = time.perf_counter() - start_total

        throughput = ThroughputMetrics(
            total_requests=THROUGHPUT_REQUESTS,
            total_duration_seconds=total_duration,
        )

        performance_report.add_throughput_result("sequential", latency_metrics, throughput)

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
        pregenerated_data: dict[str, Any],
        performance_report: PerformanceReport,
        workers: int,
    ) -> None:
        """
        Measure throughput for concurrent requests.

        Args:
            client: Test client
            pregenerated_data: Pre-generated embeddings
            performance_report: Report collector
            workers: Number of concurrent workers
        """
        query = pregenerated_data["query"]
        requests_per_worker = THROUGHPUT_REQUESTS // workers

        latency_metrics = LatencyMetrics()
        errors: list[str] = []

        def make_request(request_id: int) -> float:
            """Make a single request and return latency in ms."""
            start = time.perf_counter()
            response = client.post(
                "/search",
                json={"embedding": query, "k": 5},
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

        performance_report.add_throughput_result(
            f"concurrent_{workers}", latency_metrics, throughput
        )

        print(f"\n{'=' * 60}")
        print(f"Concurrent Throughput Test ({workers} workers)")
        print(f"{'=' * 60}")
        print(throughput.summary())
        print("\nLatency Stats:")
        print(latency_metrics.summary())
