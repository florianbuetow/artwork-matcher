"""
Metrics collection utilities for performance testing.

Provides the LatencyMetrics, ThroughputMetrics, and PerformanceReport classes
for collecting, analyzing, and reporting performance data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LatencyMetrics:
    """
    Collects and analyzes latency samples.

    Provides statistical summaries including mean, min, max,
    percentiles, and histogram distribution.
    """

    samples: list[float] = field(default_factory=list)

    def add(self, sample: float) -> None:
        """Add a latency sample in milliseconds."""
        self.samples.append(sample)

    @property
    def count(self) -> int:
        """Number of samples collected."""
        return len(self.samples)

    @property
    def mean(self) -> float:
        """Average latency in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.mean(self.samples))

    @property
    def min(self) -> float:
        """Minimum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.min(self.samples))

    @property
    def max(self) -> float:
        """Maximum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.max(self.samples))

    @property
    def std(self) -> float:
        """Standard deviation in milliseconds."""
        if not self.samples:
            return 0.0
        return float(np.std(self.samples))

    def percentile(self, p: int) -> float:
        """
        Calculate percentile value.

        Args:
            p: Percentile (0-100), e.g., 50 for median, 95 for p95

        Returns:
            Latency value at the given percentile in milliseconds
        """
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, p))

    @property
    def p50(self) -> float:
        """Median (50th percentile) in milliseconds."""
        return self.percentile(50)

    @property
    def p95(self) -> float:
        """95th percentile in milliseconds."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile in milliseconds."""
        return self.percentile(99)

    def histogram(self, bins: int = 10) -> dict[str, int]:
        """
        Generate histogram of latency distribution.

        Args:
            bins: Number of bins for the histogram

        Returns:
            Dictionary mapping bin ranges (as strings) to counts
        """
        if not self.samples:
            return {}

        counts: NDArray[np.intp]
        bin_edges: NDArray[np.floating[object]]
        counts, bin_edges = np.histogram(self.samples, bins=bins)

        result: dict[str, int] = {}
        for i, count in enumerate(counts):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            key = f"{low:.1f}-{high:.1f}ms"
            result[key] = int(count)

        return result

    def summary(self) -> str:
        """
        Generate formatted summary string for console output.

        Returns:
            Multi-line string with all metrics
        """
        if not self.samples:
            return "No samples collected"

        lines = [
            f"Samples: {self.count}",
            f"Mean:    {self.mean:.2f} ms",
            f"Min:     {self.min:.2f} ms",
            f"Max:     {self.max:.2f} ms",
            f"Std:     {self.std:.2f} ms",
            f"P50:     {self.p50:.2f} ms",
            f"P95:     {self.p95:.2f} ms",
            f"P99:     {self.p99:.2f} ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float | int]:
        """
        Convert metrics to dictionary format.

        Returns:
            Dictionary with all metric values
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class ThroughputMetrics:
    """
    Collects and analyzes throughput measurements.

    Tracks total requests, duration, and calculates requests per second.
    """

    total_requests: int = 0
    total_duration_seconds: float = 0.0

    @property
    def requests_per_second(self) -> float:
        """Calculate throughput in requests per second."""
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.total_requests / self.total_duration_seconds

    def summary(self) -> str:
        """Generate formatted summary string."""
        return (
            f"Requests: {self.total_requests}\n"
            f"Duration: {self.total_duration_seconds:.2f} s\n"
            f"Throughput: {self.requests_per_second:.2f} req/s"
        )

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary format."""
        return {
            "total_requests": self.total_requests,
            "total_duration_seconds": self.total_duration_seconds,
            "requests_per_second": self.requests_per_second,
        }


@dataclass
class PerformanceReport:
    """Collects storage service performance test results and generates markdown report."""

    object_size_results: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    throughput_results: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_object_size_result(
        self,
        operation: str,
        size_bytes: int,
        latency: LatencyMetrics,
    ) -> None:
        """Record an object size latency result for a specific operation."""
        operation_key = operation.lower()
        if operation_key not in self.object_size_results:
            self.object_size_results[operation_key] = {}
        self.object_size_results[operation_key][str(size_bytes)] = latency.to_dict()

    def add_throughput_result(
        self,
        name: str,
        latency: LatencyMetrics,
        throughput: ThroughputMetrics,
    ) -> None:
        """Record throughput test result."""
        self.throughput_results[name] = {
            **latency.to_dict(),
            **throughput.to_dict(),
        }

    def generate_markdown(self) -> str:
        """Generate full markdown report."""
        lines = [
            "# Storage Service Performance Test Report",
            "",
        ]

        lines.extend(self._generate_key_findings())
        lines.extend(self._generate_test_configuration())
        lines.extend(self._generate_object_size_section())
        lines.extend(self._generate_throughput_section())
        lines.extend(self._generate_bandwidth_section())
        lines.extend(
            [
                "---",
                "",
                "*Report generated by storage service performance tests.*",
            ]
        )

        return "\n".join(lines)

    def _generate_key_findings(self) -> list[str]:
        """Generate key findings summary."""
        lines = ["## Key Findings", ""]
        findings = []

        for operation in ("put", "get", "delete"):
            operation_results = self.object_size_results.get(operation, {})
            if not operation_results:
                continue
            sizes = sorted((int(size) for size in operation_results), key=int)
            if len(sizes) < 2:
                continue
            smallest_metrics = operation_results[str(sizes[0])]
            largest_metrics = operation_results[str(sizes[-1])]
            smallest_mean = smallest_metrics["mean"]
            largest_mean = largest_metrics["mean"]
            ratio = largest_mean / smallest_mean if smallest_mean > 0 else 0
            findings.append(
                f"- **{operation.upper()} size impact**: {self._format_size(sizes[-1])} takes "
                f"{ratio:.1f}x longer than {self._format_size(sizes[0])} "
                f"({largest_mean:.2f} ms vs {smallest_mean:.2f} ms)"
            )

        put_sequential = self.throughput_results.get("put_sequential")
        if put_sequential:
            findings.append(
                f"- **PUT sequential throughput**: "
                f"{put_sequential['requests_per_second']:.1f} req/s"
            )

        get_sequential = self.throughput_results.get("get_sequential")
        if get_sequential:
            findings.append(
                f"- **GET sequential throughput**: "
                f"{get_sequential['requests_per_second']:.1f} req/s"
            )

        concurrent = {
            name: metrics
            for name, metrics in self.throughput_results.items()
            if "concurrent" in name
        }
        if concurrent:
            best_name, best_metrics = max(
                concurrent.items(),
                key=lambda item: item[1]["requests_per_second"],
            )
            parts = best_name.split("_")
            operation = parts[0].upper() if parts else "UNKNOWN"
            workers = parts[-1] if parts and parts[-1].isdigit() else "?"
            findings.append(
                f"- **Best concurrent throughput**: "
                f"{best_metrics['requests_per_second']:.1f} req/s "
                f"({operation}, {workers} workers)"
            )

        if put_sequential and get_sequential:
            put_rps = put_sequential["requests_per_second"]
            get_rps = get_sequential["requests_per_second"]
            ratio = get_rps / put_rps if put_rps > 0 else 0
            findings.append(f"- **PUT vs GET**: Reads are {ratio:.1f}x faster than writes")

        if findings:
            lines.extend(findings)
        else:
            lines.append("*No test results available.*")

        lines.extend(["", ""])
        return lines

    def _generate_test_configuration(self) -> list[str]:
        """Generate test configuration section."""
        return [
            "## Test Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            "| Iterations per scenario | 30 |",
            "| Object sizes | 10 KB, 50 KB, 100 KB, 500 KB, 1 MB, 2 MB, 5 MB |",
            "| Throughput requests | 250 |",
            "| Throughput object size | 500 KB |",
            "| Concurrency levels | 2, 4, 8, 16 workers |",
            "",
        ]

    def _generate_object_size_section(self) -> list[str]:
        """Generate object size tests section with analysis."""
        if not self.object_size_results:
            return []

        lines = [
            "## Object Size Tests",
            "",
            "Measures how latency changes with object size for PUT, GET, and DELETE operations.",
            "",
        ]

        for operation in ("put", "get", "delete"):
            results = self.object_size_results.get(operation, {})
            if not results:
                continue

            lines.extend(
                [
                    f"### {operation.upper()} Latency by Object Size",
                    "",
                    "| Size | Mean | Std | P50 | P95 | P99 |",
                    "|------|------|-----|-----|-----|-----|",
                ]
            )

            sizes = sorted((int(size) for size in results), key=int)
            for size in sizes:
                metrics = results[str(size)]
                lines.append(
                    f"| {self._format_size(size)} | {metrics['mean']:.2f} ms | "
                    f"{metrics['std']:.2f} ms | {metrics['p50']:.2f} ms | "
                    f"{metrics['p95']:.2f} ms | {metrics['p99']:.2f} ms |"
                )

            lines.extend(["", "**Analysis:**", ""])

            if len(sizes) >= 2:
                smallest_size = sizes[0]
                largest_size = sizes[-1]
                smallest_mean = results[str(smallest_size)]["mean"]
                largest_mean = results[str(largest_size)]["mean"]
                ratio = largest_mean / smallest_mean if smallest_mean > 0 else 0
                lines.append(
                    f"- Size scaling: {self._format_size(largest_size)} takes {ratio:.1f}x longer "
                    f"than {self._format_size(smallest_size)}"
                )

            bandwidth_entries = []
            for size in sizes:
                mean_ms = results[str(size)]["mean"]
                mbps = self._to_mbps(size, mean_ms)
                bandwidth_entries.append(f"{self._format_size(size)}: {mbps:.2f} MB/s")
            if bandwidth_entries:
                lines.append("- Effective bandwidth: " + ", ".join(bandwidth_entries))

            std_values = [results[str(size)]["std"] for size in sizes]
            if std_values:
                avg_std = sum(std_values) / len(std_values)
                if avg_std < 1.0:
                    lines.append(
                        "- Consistency: highly consistent latency (low standard deviation)"
                    )
                elif avg_std < 5.0:
                    lines.append("- Consistency: consistent latency across samples")
                else:
                    lines.append(
                        f"- Consistency: notable variance in latency (avg std: {avg_std:.2f} ms)"
                    )

            lines.extend(["", ""])

        lines.extend(self._generate_cross_operation_comparison())
        return lines

    def _generate_throughput_section(self) -> list[str]:
        """Generate throughput tests section with analysis."""
        if not self.throughput_results:
            return []

        lines = [
            "## Throughput Tests",
            "",
            "Measures sustained request handling capacity.",
            "",
            "| Test | Requests | Duration | Throughput | Mean | P99 |",
            "|------|----------|----------|------------|------|-----|",
        ]

        for name, metrics in self.throughput_results.items():
            lines.append(
                f"| {name} | {metrics['total_requests']:.0f} | "
                f"{metrics['total_duration_seconds']:.2f} s | "
                f"**{metrics['requests_per_second']:.1f} req/s** | "
                f"{metrics['mean']:.2f} ms | {metrics['p99']:.2f} ms |"
            )

        lines.extend(["", "**Analysis:**", ""])

        if "put_sequential" in self.throughput_results:
            put_seq = self.throughput_results["put_sequential"]
            lines.append(f"- Sequential PUT baseline: {put_seq['requests_per_second']:.1f} req/s")

        if "get_sequential" in self.throughput_results:
            get_seq = self.throughput_results["get_sequential"]
            lines.append(f"- Sequential GET baseline: {get_seq['requests_per_second']:.1f} req/s")

        for operation in ("put", "get"):
            seq_key = f"{operation}_sequential"
            if seq_key not in self.throughput_results:
                continue

            seq_rps = self.throughput_results[seq_key]["requests_per_second"]
            concurrent_keys = sorted(
                (
                    key
                    for key in self.throughput_results
                    if key.startswith(f"{operation}_concurrent_")
                ),
                key=self._worker_count,
            )
            for key in concurrent_keys:
                metrics = self.throughput_results[key]
                speedup = metrics["requests_per_second"] / seq_rps if seq_rps > 0 else 0
                if speedup > 1.5:
                    lines.append(
                        f"- {key}: **{speedup:.2f}x speedup** over {seq_key} (good parallelization)"
                    )
                elif speedup > 1.1:
                    lines.append(f"- {key}: {speedup:.2f}x speedup over {seq_key}")
                elif speedup < 0.9:
                    lines.append(f"- {key}: **contention detected** ({speedup:.2f}x of {seq_key})")
                else:
                    lines.append(f"- {key}: no significant speedup (likely CPU-bound)")

        put_seq = self.throughput_results.get("put_sequential")
        get_seq = self.throughput_results.get("get_sequential")
        if put_seq and get_seq:
            put_rps = put_seq["requests_per_second"]
            get_rps = get_seq["requests_per_second"]
            ratio = get_rps / put_rps if put_rps > 0 else 0
            lines.append(
                f"- PUT vs GET sequential throughput: reads are {ratio:.2f}x faster than writes"
            )

        lines.extend(["", ""])
        return lines

    def _generate_bandwidth_section(self) -> list[str]:
        """Generate effective bandwidth section from object size results."""
        put_results = self.object_size_results.get("put")
        get_results = self.object_size_results.get("get")
        if not put_results and not get_results:
            return []

        candidate_sizes = set()
        if put_results:
            candidate_sizes.update(int(size) for size in put_results)
        if get_results:
            candidate_sizes.update(int(size) for size in get_results)

        if not candidate_sizes:
            return []

        sizes = sorted(candidate_sizes)
        lines = [
            "## Effective Bandwidth",
            "",
            "Derived throughput from mean latency across object sizes.",
            "",
            "| Size | PUT MB/s | GET MB/s |",
            "|------|----------|----------|",
        ]

        put_bandwidth: list[float] = []
        get_bandwidth: list[float] = []

        for size in sizes:
            put_mbps = 0.0
            get_mbps = 0.0

            if put_results and str(size) in put_results:
                put_mbps = self._to_mbps(size, put_results[str(size)]["mean"])
                put_bandwidth.append(put_mbps)

            if get_results and str(size) in get_results:
                get_mbps = self._to_mbps(size, get_results[str(size)]["mean"])
                get_bandwidth.append(get_mbps)

            lines.append(f"| {self._format_size(size)} | {put_mbps:.2f} | {get_mbps:.2f} |")

        lines.extend(["", "**Analysis:**", ""])

        if len(put_bandwidth) >= 2:
            put_plateau = self._detect_plateau(put_bandwidth)
            lines.append(f"- PUT bandwidth trend: {put_plateau}")

        if len(get_bandwidth) >= 2:
            get_plateau = self._detect_plateau(get_bandwidth)
            lines.append(f"- GET bandwidth trend: {get_plateau}")

        if put_bandwidth and get_bandwidth:
            avg_put = sum(put_bandwidth) / len(put_bandwidth)
            avg_get = sum(get_bandwidth) / len(get_bandwidth)
            ratio = avg_get / avg_put if avg_put > 0 else 0
            if ratio >= 1.0:
                lines.append(
                    f"- Read vs write bandwidth: reads average {ratio:.2f}x higher throughput"
                )
            else:
                inverse = avg_put / avg_get if avg_get > 0 else 0
                lines.append(
                    f"- Read vs write bandwidth: writes average {inverse:.2f}x higher throughput"
                )

        lines.extend(["", ""])
        return lines

    def _generate_cross_operation_comparison(self) -> list[str]:
        """Generate cross-operation comparison for the largest shared size."""
        lines = [
            "### Cross-Operation Comparison",
            "",
        ]

        put_results = self.object_size_results.get("put", {})
        if not put_results:
            lines.extend(["*No cross-operation comparison available.*", "", ""])
            return lines

        put_sizes = {int(size) for size in put_results}
        get_sizes = {int(size) for size in self.object_size_results.get("get", {})}
        delete_sizes = {int(size) for size in self.object_size_results.get("delete", {})}

        put_get_shared = sorted(put_sizes & get_sizes)
        if put_get_shared:
            size = put_get_shared[-1]
            put_mean = put_results[str(size)]["mean"]
            get_mean = self.object_size_results["get"][str(size)]["mean"]
            ratio = put_mean / get_mean if get_mean > 0 else 0
            lines.append(
                f"- At {self._format_size(size)}, GET is {ratio:.2f}x faster than PUT "
                f"({get_mean:.2f} ms vs {put_mean:.2f} ms)"
            )

        put_delete_shared = sorted(put_sizes & delete_sizes)
        if put_delete_shared:
            size = put_delete_shared[-1]
            put_mean = put_results[str(size)]["mean"]
            delete_mean = self.object_size_results["delete"][str(size)]["mean"]
            ratio = put_mean / delete_mean if delete_mean > 0 else 0
            lines.append(
                f"- At {self._format_size(size)}, DELETE is {ratio:.2f}x faster than PUT "
                f"({delete_mean:.2f} ms vs {put_mean:.2f} ms)"
            )

        if len(lines) == 2:
            lines.append("*No cross-operation comparison available.*")

        lines.extend(["", ""])
        return lines

    @staticmethod
    def _worker_count(name: str) -> int:
        """Extract worker count from throughput test name."""
        parts = name.split("_")
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return 0

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in bytes to human-readable units."""
        if size_bytes >= 1_048_576:
            return f"{size_bytes / 1_048_576:.0f} MB"
        return f"{size_bytes / 1_024:.0f} KB"

    @staticmethod
    def _to_mbps(size_bytes: int, mean_ms: float) -> float:
        """Convert size and latency to effective MB/s."""
        if mean_ms <= 0:
            return 0.0
        return size_bytes / (mean_ms / 1000.0) / 1_048_576

    @staticmethod
    def _detect_plateau(values: list[float]) -> str:
        """Detect if values plateau at larger sizes."""
        if len(values) < 3:
            return "insufficient data for plateau detection"

        early_growth = values[1] - values[0]
        late_growth = values[-1] - values[-2]
        if early_growth <= 0:
            return "bandwidth decreases at larger sizes"

        growth_ratio = late_growth / early_growth
        if growth_ratio < 0.25:
            return "bandwidth appears to plateau at larger object sizes"
        if growth_ratio < 0:
            return "bandwidth regresses at larger object sizes"
        return "bandwidth continues scaling with object size"
