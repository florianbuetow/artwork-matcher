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
    """Collects search service performance test results and generates markdown report."""

    index_size_results: dict[str, dict[str, float]] = field(default_factory=dict)
    k_value_results: dict[str, dict[str, float]] = field(default_factory=dict)
    throughput_results: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_index_size_result(self, size: int, latency: LatencyMetrics) -> None:
        """Record index size test result."""
        key = str(size)
        self.index_size_results[key] = latency.to_dict()

    def add_k_value_result(self, k: int, latency: LatencyMetrics) -> None:
        """Record k value test result."""
        key = f"k={k}"
        self.k_value_results[key] = latency.to_dict()

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
        """Generate markdown report content with insights."""
        lines = [
            "# Search Service Performance Test Report",
            "",
        ]

        lines.extend(self._generate_key_findings())
        lines.extend(self._generate_test_configuration())

        if self.index_size_results:
            lines.extend(self._generate_index_size_section())

        if self.k_value_results:
            lines.extend(self._generate_k_value_section())

        if self.throughput_results:
            lines.extend(self._generate_throughput_section())

        lines.extend(
            [
                "---",
                "",
                "*Report generated by search service performance tests.*",
            ]
        )

        return "\n".join(lines)

    def _generate_key_findings(self) -> list[str]:
        """Generate key findings summary."""
        lines = ["## Key Findings", ""]
        findings = []

        if self.index_size_results:
            sizes = list(self.index_size_results.keys())
            if len(sizes) >= 2:
                smallest = self.index_size_results[sizes[0]]
                largest = self.index_size_results[sizes[-1]]
                smallest_size = int(sizes[0])
                largest_size = int(sizes[-1])
                latency_ratio = largest["mean"] / smallest["mean"] if smallest["mean"] > 0 else 0

                findings.append(
                    f"- **Index size impact**: {largest_size:,} vectors take "
                    f"{latency_ratio:.1f}x longer than {smallest_size:,} vectors "
                    f"({largest['mean']:.2f} ms vs {smallest['mean']:.2f} ms)"
                )

        if self.k_value_results:
            k_keys = list(self.k_value_results.keys())
            k_values = list(self.k_value_results.values())
            if len(k_values) >= 2:
                first = k_values[0]
                last = k_values[-1]
                ratio = last["mean"] / first["mean"] if first["mean"] > 0 else 0
                findings.append(
                    f"- **K value impact**: {k_keys[-1]} takes "
                    f"{ratio:.1f}x longer than {k_keys[0]} "
                    f"({last['mean']:.2f} ms vs {first['mean']:.2f} ms)"
                )

        if self.throughput_results:
            if "sequential" in self.throughput_results:
                seq = self.throughput_results["sequential"]
                findings.append(
                    f"- **Sequential throughput**: {seq['requests_per_second']:.1f} req/s"
                )

            concurrent = {k: v for k, v in self.throughput_results.items() if k != "sequential"}
            if concurrent:
                best = max(concurrent.items(), key=lambda x: x[1]["requests_per_second"])
                findings.append(
                    f"- **Best concurrent throughput**: "
                    f"{best[1]['requests_per_second']:.1f} req/s ({best[0]})"
                )

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
            "| Iterations per test | 30 |",
            "| Index type | IndexFlatIP (exhaustive search) |",
            "| Embedding dimension | 768 |",
            "| Distance metric | Inner product (cosine on normalized vectors) |",
            "",
        ]

    def _generate_index_size_section(self) -> list[str]:
        """Generate index size tests section with analysis."""
        lines = [
            "## Index Size Tests",
            "",
            "Measures how search latency scales with the number of vectors in the index. "
            "Flat index performs exhaustive search, so O(n) complexity is expected.",
            "",
            "| Index Size | Mean | Std | P50 | P95 | P99 |",
            "|------------|------|-----|-----|-----|-----|",
        ]

        for size, metrics in self.index_size_results.items():
            lines.append(
                f"| {int(size):,} | {metrics['mean']:.2f} ms | {metrics['std']:.2f} ms | "
                f"{metrics['p50']:.2f} ms | {metrics['p95']:.2f} ms | "
                f"{metrics['p99']:.2f} ms |"
            )

        # Analysis
        lines.extend(["", "**Analysis:**", ""])
        sizes = list(self.index_size_results.keys())
        if len(sizes) >= 2:
            first = self.index_size_results[sizes[0]]
            last = self.index_size_results[sizes[-1]]
            first_size = int(sizes[0])
            last_size = int(sizes[-1])
            ratio = last["mean"] / first["mean"] if first["mean"] > 0 else 0
            size_ratio = last_size / first_size

            # Calculate ms per 1000 vectors
            ms_per_1k = (last["mean"] - first["mean"]) / (last_size - first_size) * 1000
            lines.append(
                f"- Scaling from {first_size:,} to {last_size:,} vectors adds "
                f"~{last['mean'] - first['mean']:.2f} ms (~{ms_per_1k:.3f} ms per 1,000 vectors)"
            )

            # Check if scaling is sub-linear (better than O(n))
            if ratio < size_ratio * 0.5:
                lines.append(
                    f"- Scaling is **sub-linear** "
                    f"({ratio:.1f}x for {size_ratio:.0f}x more vectors) "
                    "- FAISS optimizations are effective"
                )
            elif abs(ratio - size_ratio) / size_ratio < 0.3:
                lines.append(
                    "- Latency scales approximately **linearly** with index size "
                    "(confirms O(n) complexity)"
                )
            else:
                lines.append(f"- Scaling factor: {ratio:.1f}x for {size_ratio:.0f}x more vectors")

            # Check consistency (low std = consistent)
            avg_std = sum(m["std"] for m in self.index_size_results.values()) / len(sizes)
            if avg_std < 0.5:
                lines.append("- Latency is **highly consistent** (very low standard deviation)")
            elif avg_std < 2.0:
                lines.append("- Latency is **consistent** (low standard deviation)")
            else:
                lines.append(f"- Latency variance is notable (avg std: {avg_std:.2f} ms)")

        lines.extend(["", ""])
        return lines

    def _generate_k_value_section(self) -> list[str]:
        """Generate k value tests section with analysis."""
        lines = [
            "## K Value Tests",
            "",
            "Measures whether the number of requested results affects search latency. "
            "For flat index, k should have minimal impact since all distances are computed.",
            "",
            "| K Value | Mean | Std | P50 | P95 | P99 |",
            "|---------|------|-----|-----|-----|-----|",
        ]

        for k_str, metrics in self.k_value_results.items():
            lines.append(
                f"| {k_str} | {metrics['mean']:.2f} ms | {metrics['std']:.2f} ms | "
                f"{metrics['p50']:.2f} ms | {metrics['p95']:.2f} ms | "
                f"{metrics['p99']:.2f} ms |"
            )

        # Analysis
        lines.extend(["", "**Analysis:**", ""])
        k_keys = list(self.k_value_results.keys())
        k_values = list(self.k_value_results.values())
        if len(k_values) >= 2:
            first = k_values[0]
            last = k_values[-1]
            ratio = last["mean"] / first["mean"] if first["mean"] > 0 else 0
            overhead = last["mean"] - first["mean"]

            if ratio < 1.1:
                lines.append(
                    f"- K value has **minimal impact** on latency "
                    f"(only {overhead:.2f} ms difference between {k_keys[0]} and {k_keys[-1]})"
                )
            elif ratio < 1.5:
                lines.append(
                    f"- K value has **moderate impact**: "
                    f"{k_keys[-1]} adds ~{overhead:.2f} ms over {k_keys[0]}"
                )
            else:
                lines.append(
                    f"- K value has **significant impact**: "
                    f"{ratio:.1f}x slowdown from {k_keys[0]} to {k_keys[-1]}"
                )

            # Check consistency
            avg_std = sum(m["std"] for m in k_values) / len(k_values)
            if avg_std < 0.5:
                lines.append("- Latency is **highly consistent** across all k values")
            elif avg_std < 2.0:
                lines.append("- Latency is **consistent** across k values")

        lines.extend(["", ""])
        return lines

    def _generate_throughput_section(self) -> list[str]:
        """Generate throughput tests section with analysis."""
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

        # Analysis
        lines.extend(["", "**Analysis:**", ""])

        if "sequential" in self.throughput_results:
            seq = self.throughput_results["sequential"]
            lines.append(f"- Sequential baseline: {seq['requests_per_second']:.1f} req/s")

        concurrent = {k: v for k, v in self.throughput_results.items() if k != "sequential"}
        if concurrent and "sequential" in self.throughput_results:
            seq_rps = self.throughput_results["sequential"]["requests_per_second"]
            for name, metrics in concurrent.items():
                speedup = metrics["requests_per_second"] / seq_rps if seq_rps > 0 else 0
                if speedup > 1.5:
                    lines.append(
                        f"- {name}: **{speedup:.2f}x speedup** over sequential "
                        f"(good parallelization)"
                    )
                elif speedup > 1.1:
                    lines.append(f"- {name}: {speedup:.2f}x speedup over sequential")
                elif speedup < 0.9:
                    lines.append(
                        f"- {name}: **contention detected** ({speedup:.2f}x of sequential)"
                    )
                else:
                    lines.append(f"- {name}: no significant speedup (likely CPU-bound)")

            # Summary insight
            best = max(concurrent.items(), key=lambda x: x[1]["requests_per_second"])
            best_speedup = best[1]["requests_per_second"] / seq_rps if seq_rps > 0 else 0
            if best_speedup > 1.2:
                lines.append(
                    f"- **Recommendation**: Use concurrent requests "
                    f"for {best_speedup:.1f}x throughput gain"
                )
            else:
                lines.append(
                    "- **Recommendation**: Service is CPU-bound, "
                    "concurrency provides limited benefit"
                )

        lines.extend(["", ""])
        return lines
