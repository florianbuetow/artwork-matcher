"""
Metrics collection utilities for performance testing.

Provides the LatencyMetrics class for collecting and analyzing
request timing data with statistical summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    """
    Collects performance test results and generates markdown report.
    """

    dimension_results: dict[str, dict[str, float]] = field(default_factory=dict)
    filesize_results: dict[str, dict[str, float]] = field(default_factory=dict)
    throughput_results: dict[str, dict[str, float]] = field(default_factory=dict)
    image_sizes: dict[str, float] = field(default_factory=dict)

    def add_dimension_result(
        self, size: int, latency: LatencyMetrics, image_size_kb: float
    ) -> None:
        """Record dimension test result."""
        key = f"{size}x{size}"
        self.dimension_results[key] = latency.to_dict()
        self.image_sizes[f"dim_{key}"] = image_size_kb

    def add_filesize_result(
        self, target_kb: int, latency: LatencyMetrics, actual_size_kb: float
    ) -> None:
        """Record file size test result."""
        key = f"{target_kb}kb"
        self.filesize_results[key] = {
            **latency.to_dict(),
            "target_kb": target_kb,
            "actual_kb": actual_size_kb,
        }

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
            "# Embeddings Service Performance Test Report",
            "",
            f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
        ]

        # Key findings summary
        lines.extend(self._generate_key_findings())

        # Test configuration
        lines.extend(
            [
                "## Test Configuration",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
                "| Iterations per test | 30 |",
                "| Model | DINOv2 (facebook/dinov2-base) |",
                "| Input size | 518x518 (model native) |",
                "",
            ]
        )

        # Dimension tests
        if self.dimension_results:
            lines.extend(self._generate_dimension_section())

        # File size tests
        if self.filesize_results:
            lines.extend(self._generate_filesize_section())

        # Throughput tests
        if self.throughput_results:
            lines.extend(self._generate_throughput_section())

        lines.extend(
            [
                "---",
                "",
                "*Report generated by embeddings service performance tests.*",
            ]
        )

        return "\n".join(lines)

    def _generate_key_findings(self) -> list[str]:
        """Generate key findings summary."""
        lines = ["## Key Findings", ""]
        findings = []

        if self.dimension_results:
            # Find baseline (smallest) and largest
            dims = list(self.dimension_results.keys())
            if len(dims) >= 2:
                smallest = self.dimension_results[dims[0]]
                largest = self.dimension_results[dims[-1]]
                ratio = largest["mean"] / smallest["mean"] if smallest["mean"] > 0 else 0
                findings.append(
                    f"- **Image dimension impact**: {dims[-1]} images take "
                    f"{ratio:.1f}x longer than {dims[0]} images "
                    f"({largest['mean']:.1f} ms vs {smallest['mean']:.1f} ms)"
                )

        if self.filesize_results:
            sizes = list(self.filesize_results.values())
            if len(sizes) >= 2:
                smallest = sizes[0]
                largest = sizes[-1]
                ratio = largest["mean"] / smallest["mean"] if smallest["mean"] > 0 else 0
                findings.append(
                    f"- **File size impact**: {largest['actual_kb']:.0f} KB files take "
                    f"{ratio:.1f}x longer than {smallest['actual_kb']:.0f} KB files"
                )

        if self.throughput_results:
            if "sequential" in self.throughput_results:
                seq = self.throughput_results["sequential"]
                findings.append(
                    f"- **Sequential throughput**: {seq['requests_per_second']:.1f} req/s"
                )

            # Find best concurrent result
            concurrent_results = {
                k: v for k, v in self.throughput_results.items() if k != "sequential"
            }
            if concurrent_results:
                best = max(concurrent_results.items(), key=lambda x: x[1]["requests_per_second"])
                findings.append(
                    f"- **Best concurrent throughput**: {best[1]['requests_per_second']:.1f} req/s "
                    f"({best[0]})"
                )

        if findings:
            lines.extend(findings)
        else:
            lines.append("*No test results available.*")

        lines.extend(["", ""])
        return lines

    def _generate_dimension_section(self) -> list[str]:
        """Generate dimension tests section with analysis."""
        lines = [
            "## Image Dimension Tests",
            "",
            "Measures preprocessing overhead for different input sizes. "
            "All images are resized to 518x518 before model inference.",
            "",
            "| Dimensions | File Size | Mean | Std | P50 | P95 | P99 |",
            "|------------|-----------|------|-----|-----|-----|-----|",
        ]

        for dim, metrics in self.dimension_results.items():
            img_key = f"dim_{dim}"
            img_size = self.image_sizes.get(img_key, 0)
            lines.append(
                f"| {dim} | {img_size:.0f} KB | "
                f"{metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        # Analysis
        lines.extend(["", "**Analysis:**", ""])
        dims = list(self.dimension_results.keys())
        if len(dims) >= 2:
            first = self.dimension_results[dims[0]]
            last = self.dimension_results[dims[-1]]
            overhead = last["mean"] - first["mean"]
            lines.append(f"- Resizing from {dims[-1]} to 518x518 adds ~{overhead:.1f} ms overhead")

            # Check consistency (low std = consistent)
            avg_std = sum(m["std"] for m in self.dimension_results.values()) / len(dims)
            if avg_std < 5:
                lines.append("- Latency is **consistent** (low standard deviation)")
            else:
                lines.append(f"- Latency variance is notable (avg std: {avg_std:.1f} ms)")

        lines.extend(["", ""])
        return lines

    def _generate_filesize_section(self) -> list[str]:
        """Generate file size tests section with analysis."""
        lines = [
            "## File Size Tests",
            "",
            "Measures impact of JPEG file complexity on processing time. "
            "Uses noise images to control compressed size.",
            "",
            "| Target | Actual | Mean | Std | P50 | P95 | P99 |",
            "|--------|--------|------|-----|-----|-----|-----|",
        ]

        for _, metrics in self.filesize_results.items():
            lines.append(
                f"| {metrics['target_kb']:.0f} KB | {metrics['actual_kb']:.0f} KB | "
                f"{metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        # Analysis
        lines.extend(["", "**Analysis:**", ""])
        sizes = list(self.filesize_results.values())
        if len(sizes) >= 2:
            ratio = sizes[-1]["mean"] / sizes[0]["mean"] if sizes[0]["mean"] > 0 else 0
            size_ratio = sizes[-1]["actual_kb"] / sizes[0]["actual_kb"]
            if ratio < size_ratio * 0.5:
                lines.append(
                    "- File size has **minimal impact** on latency (decoding is not the bottleneck)"
                )
            else:
                lines.append(
                    f"- File size has **moderate impact**: "
                    f"{size_ratio:.0f}x larger files → {ratio:.1f}x slower"
                )

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
                f"{metrics['mean']:.1f} ms | {metrics['p99']:.1f} ms |"
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
                if speedup > 1.1:
                    lines.append(f"- {name}: {speedup:.2f}x speedup over sequential")
                elif speedup < 0.9:
                    lines.append(
                        f"- {name}: **contention detected** ({speedup:.2f}x of sequential)"
                    )
                else:
                    lines.append(f"- {name}: no significant speedup (likely CPU-bound)")

        lines.extend(["", ""])
        return lines
