"""
Metrics collection utilities for performance testing.

Provides the LatencyMetrics and ThroughputMetrics classes for collecting
and analyzing request timing data with statistical summaries, and
PerformanceReport for generating comprehensive markdown reports.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


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
        return statistics.mean(self.samples)

    @property
    def min(self) -> float:
        """Minimum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return min(self.samples)

    @property
    def max(self) -> float:
        """Maximum latency in milliseconds."""
        if not self.samples:
            return 0.0
        return max(self.samples)

    @property
    def std(self) -> float:
        """Standard deviation in milliseconds."""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)

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
        sorted_samples = sorted(self.samples)
        index = int(len(sorted_samples) * p / 100)
        index = min(index, len(sorted_samples) - 1)
        return sorted_samples[index]

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

    Provides comprehensive analysis with key findings, organized test sections,
    and detailed analysis similar to the embeddings service report.
    """

    # Image dimension tests (varying input image sizes)
    dimension_results: dict[str, dict[str, float]] = field(default_factory=dict)
    dimension_image_sizes: dict[str, float] = field(default_factory=dict)

    # Endpoint latency tests (different endpoints)
    endpoint_results: dict[str, dict[str, float]] = field(default_factory=dict)

    # Pipeline options tests (with/without geometric verification)
    options_results: dict[str, dict[str, float]] = field(default_factory=dict)

    # Throughput tests (sequential and concurrent)
    throughput_results: dict[str, dict[str, float]] = field(default_factory=dict)

    # Geometric service status
    geometric_service_status: dict[str, str | bool] = field(default_factory=dict)

    def add_dimension_result(
        self, size: int, latency: LatencyMetrics, image_size_kb: float
    ) -> None:
        """Record image dimension test result."""
        key = f"{size}x{size}"
        self.dimension_results[key] = latency.to_dict()
        self.dimension_image_sizes[key] = image_size_kb

    def add_endpoint_result(self, endpoint: str, latency: LatencyMetrics) -> None:
        """Record endpoint latency test result."""
        self.endpoint_results[endpoint] = latency.to_dict()

    def add_options_result(self, name: str, latency: LatencyMetrics) -> None:
        """Record pipeline options test result."""
        self.options_results[name] = latency.to_dict()

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

    def set_geometric_service_status(self, available: bool, message: str) -> None:
        """Record geometric service availability status."""
        self.geometric_service_status = {
            "available": available,
            "message": message,
        }

    def generate_markdown(self) -> str:
        """Generate markdown report content with comprehensive analysis."""
        lines = [
            "# Gateway Service Performance Test Report",
            "",
        ]

        # Key findings summary
        lines.extend(self._generate_key_findings())

        # Test configuration
        lines.extend(self._generate_config_section())

        # Image dimension tests
        if self.dimension_results:
            lines.extend(self._generate_dimension_section())

        # Endpoint latency tests
        if self.endpoint_results:
            lines.extend(self._generate_endpoint_section())

        # Geometric service status
        if self.geometric_service_status:
            lines.extend(self._generate_geometric_status_section())

        # Pipeline options tests
        if self.options_results:
            lines.extend(self._generate_options_section())

        # Throughput tests
        if self.throughput_results:
            lines.extend(self._generate_throughput_section())

        lines.extend(
            [
                "---",
                "",
                "*Report generated by gateway service performance tests.*",
            ]
        )

        return "\n".join(lines)

    def _generate_key_findings(self) -> list[str]:
        """Generate key findings summary."""
        lines = ["## Key Findings", ""]
        findings: list[str] = []

        # Collect findings from each category
        findings.extend(self._get_dimension_findings())
        findings.extend(self._get_endpoint_findings())
        findings.extend(self._get_options_findings())
        findings.extend(self._get_throughput_findings())

        # Add geometric service status finding
        if self.geometric_service_status and not self.geometric_service_status.get("available"):
            message = self.geometric_service_status.get("message", "Not tested")
            findings.append(f"- **Geometric service**: {message}")

        if findings:
            lines.extend(findings)
        else:
            lines.append("*No test results available.*")

        lines.extend(["", ""])
        return lines

    def _get_dimension_findings(self) -> list[str]:
        """Get key findings for dimension tests."""
        if len(self.dimension_results) < 2:
            return []
        dims = list(self.dimension_results.keys())
        smallest = self.dimension_results[dims[0]]
        largest = self.dimension_results[dims[-1]]
        if smallest["mean"] <= 0:
            return []
        ratio = largest["mean"] / smallest["mean"]
        smallest_size = self.dimension_image_sizes.get(dims[0], 0)
        largest_size = self.dimension_image_sizes.get(dims[-1], 0)
        return [
            f"- **Image dimension impact**: {dims[-1]} images "
            f"({largest_size:.0f} KB) take {ratio:.1f}x longer than "
            f"{dims[0]} images ({smallest_size:.0f} KB)"
        ]

    def _get_endpoint_findings(self) -> list[str]:
        """Get key findings for endpoint tests."""
        findings: list[str] = []
        if not self.endpoint_results:
            return findings
        if "identify" in self.endpoint_results:
            identify_mean = self.endpoint_results["identify"]["mean"]
            findings.append(
                f"- **Identify endpoint latency**: {identify_mean:.1f} ms (gateway overhead only)"
            )
        if "health" in self.endpoint_results:
            health_mean = self.endpoint_results["health"]["mean"]
            findings.append(f"- **Health check latency**: {health_mean:.1f} ms")
        if "info" in self.endpoint_results:
            info_mean = self.endpoint_results["info"]["mean"]
            findings.append(f"- **Info endpoint latency**: {info_mean:.1f} ms")
        return findings

    def _get_options_findings(self) -> list[str]:
        """Get key findings for options tests."""
        has_both = (
            "with_geometric" in self.options_results and "without_geometric" in self.options_results
        )
        if len(self.options_results) < 2 or not has_both:
            return []
        with_geo = self.options_results["with_geometric"]["mean"]
        without_geo = self.options_results["without_geometric"]["mean"]
        diff = with_geo - without_geo
        return [
            f"- **Geometric verification overhead**: {diff:.1f} ms "
            f"({with_geo:.1f} ms vs {without_geo:.1f} ms)"
        ]

    def _get_throughput_findings(self) -> list[str]:
        """Get key findings for throughput tests."""
        findings: list[str] = []
        if not self.throughput_results:
            return findings
        if "sequential" in self.throughput_results:
            seq = self.throughput_results["sequential"]
            findings.append(f"- **Sequential throughput**: {seq['requests_per_second']:.1f} req/s")
        concurrent_results = {k: v for k, v in self.throughput_results.items() if k != "sequential"}
        if concurrent_results:
            best = max(concurrent_results.items(), key=lambda x: x[1]["requests_per_second"])
            findings.append(
                f"- **Best concurrent throughput**: "
                f"{best[1]['requests_per_second']:.1f} req/s ({best[0]})"
            )
        return findings

    def _generate_config_section(self) -> list[str]:
        """Generate test configuration section."""
        return [
            "## Test Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            "| Iterations per test | 30 |",
            "| Throughput requests | 250 |",
            "| Backend services | Mocked (instant response) |",
            "| Concurrency levels | 2, 4, 8, 16 workers |",
            "",
            "**Note:** These tests measure **gateway overhead only**. "
            "Backend services (embeddings, search, geometric) are mocked to return "
            "instant responses. Real-world latency will be higher depending on "
            "backend performance.",
            "",
        ]

    def _generate_dimension_section(self) -> list[str]:
        """Generate image dimension tests section with analysis."""
        lines = [
            "## Image Dimension Tests",
            "",
            "Measures gateway overhead when processing different image payload sizes. "
            "The gateway validates and forwards images but does not decode pixels.",
            "",
            "| Dimensions | File Size | Mean | Std | P50 | P95 | P99 |",
            "|------------|-----------|------|-----|-----|-----|-----|",
        ]

        for dim, metrics in self.dimension_results.items():
            img_size = self.dimension_image_sizes.get(dim, 0)
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
            first_size = self.dimension_image_sizes.get(dims[0], 0)
            last_size = self.dimension_image_sizes.get(dims[-1], 0)

            if first["mean"] > 0:
                latency_ratio = last["mean"] / first["mean"]
                size_ratio = last_size / first_size if first_size > 0 else 0

                if latency_ratio < 2:
                    lines.append(
                        f"- Latency scales **sub-linearly** with image size "
                        f"({size_ratio:.0f}x larger → {latency_ratio:.1f}x slower)"
                    )
                    lines.append(
                        "- Gateway overhead is primarily JSON serialization, not image size"
                    )
                else:
                    lines.append(
                        f"- Latency increases with image size: "
                        f"{size_ratio:.0f}x larger → {latency_ratio:.1f}x slower"
                    )

            # Check consistency
            avg_std = sum(m["std"] for m in self.dimension_results.values()) / len(dims)
            if avg_std < 5:
                lines.append("- Latency is **consistent** across image sizes")
            else:
                lines.append(f"- Latency variance is notable (avg std: {avg_std:.1f} ms)")

        lines.extend(["", ""])
        return lines

    def _generate_endpoint_section(self) -> list[str]:
        """Generate endpoint latency tests section with analysis."""
        lines = [
            "## Endpoint Latency Tests",
            "",
            "Measures latency for each gateway endpoint.",
            "",
            "| Endpoint | Mean | Std | P50 | P95 | P99 |",
            "|----------|------|-----|-----|-----|-----|",
        ]

        # Sort endpoints by mean latency
        sorted_endpoints = sorted(self.endpoint_results.items(), key=lambda x: x[1]["mean"])

        for endpoint, metrics in sorted_endpoints:
            lines.append(
                f"| {endpoint} | {metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        # Analysis
        lines.extend(["", "**Analysis:**", ""])

        # Group by endpoint type
        simple_endpoints = ["health", "info", "objects", "objects_by_id"]
        complex_endpoints = ["identify"]

        simple_latencies = [
            self.endpoint_results[e]["mean"] for e in simple_endpoints if e in self.endpoint_results
        ]
        complex_latencies = [
            self.endpoint_results[e]["mean"]
            for e in complex_endpoints
            if e in self.endpoint_results
        ]

        if simple_latencies:
            avg_simple = sum(simple_latencies) / len(simple_latencies)
            lines.append(
                f"- Simple endpoints (health, info, objects): ~{avg_simple:.1f} ms average"
            )

        if complex_latencies:
            avg_complex = sum(complex_latencies) / len(complex_latencies)
            lines.append(
                f"- Pipeline endpoint (identify): ~{avg_complex:.1f} ms "
                "(includes backend coordination)"
            )

        # Check for outliers
        for endpoint, metrics in self.endpoint_results.items():
            if metrics["mean"] > 0:
                p99_ratio = metrics["p99"] / metrics["mean"]
                if p99_ratio > 3:
                    lines.append(
                        f"- **{endpoint}**: P99 is {p99_ratio:.1f}x mean - investigate outliers"
                    )

        lines.extend(["", ""])
        return lines

    def _generate_geometric_status_section(self) -> list[str]:
        """Generate geometric service status section."""
        lines = ["## Geometric Service Status", ""]

        if self.geometric_service_status.get("available"):
            lines.append("Geometric service is available.")
        else:
            message = self.geometric_service_status.get("message", "Unknown")
            lines.append(f"**{message}**")
            lines.append("")
            lines.append("Geometric verification performance tests were skipped.")

        lines.extend(["", ""])
        return lines

    def _generate_options_section(self) -> list[str]:
        """Generate pipeline options tests section with analysis."""
        lines = [
            "## Pipeline Options Tests",
            "",
            "Measures impact of different pipeline configuration options.",
            "",
            "| Configuration | Mean | Std | P50 | P95 | P99 |",
            "|---------------|------|-----|-----|-----|-----|",
        ]

        for name, metrics in self.options_results.items():
            display_name = name.replace("_", " ").title()
            lines.append(
                f"| {display_name} | {metrics['mean']:.1f} ms | {metrics['std']:.1f} ms | "
                f"{metrics['p50']:.1f} ms | {metrics['p95']:.1f} ms | "
                f"{metrics['p99']:.1f} ms |"
            )

        # Analysis
        lines.extend(["", "**Analysis:**", ""])

        has_geo_options = (
            "with_geometric" in self.options_results and "without_geometric" in self.options_results
        )
        if has_geo_options:
            with_geo = self.options_results["with_geometric"]
            without_geo = self.options_results["without_geometric"]
            overhead = with_geo["mean"] - without_geo["mean"]

            if overhead > 0:
                base = without_geo["mean"]
                pct_increase = (overhead / base) * 100 if base > 0 else 0
                lines.append(
                    f"- Geometric verification adds {overhead:.1f} ms overhead "
                    f"({pct_increase:.0f}% increase)"
                )
            else:
                lines.append("- No measurable overhead from geometric verification option")

        # Check K value impact
        k_results = {k: v for k, v in self.options_results.items() if k.startswith("k_")}
        if len(k_results) >= 2:
            sorted_k = sorted(k_results.items(), key=lambda x: int(x[0].split("_")[1]))
            first_k = sorted_k[0]
            last_k = sorted_k[-1]
            k_overhead = last_k[1]["mean"] - first_k[1]["mean"]
            if k_overhead > 0.5:
                lines.append(
                    f"- Increasing K from {first_k[0]} to {last_k[0]} adds {k_overhead:.1f} ms"
                )
            else:
                lines.append(
                    f"- K value ({first_k[0]} to {last_k[0]}) "
                    "has minimal impact on gateway overhead"
                )

        lines.extend(["", ""])
        return lines

    def _generate_throughput_section(self) -> list[str]:
        """Generate throughput tests section with analysis."""
        lines = [
            "## Throughput Tests",
            "",
            "Measures sustained request handling capacity under different concurrency levels.",
            "",
            "| Test | Requests | Duration | Throughput | Mean Latency | P99 Latency |",
            "|------|----------|----------|------------|--------------|-------------|",
        ]

        # Sort: sequential first, then by concurrency level
        def sort_key(item: tuple[str, dict[str, float]]) -> tuple[int, int]:
            name = item[0]
            if name == "sequential":
                return (0, 0)
            if name.startswith("concurrent_"):
                return (1, int(name.split("_")[1]))
            return (2, 0)

        sorted_results = sorted(self.throughput_results.items(), key=sort_key)

        for name, metrics in sorted_results:
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
            seq_rps = seq["requests_per_second"]

            # Analyze each concurrency level
            for name, metrics in sorted(concurrent.items(), key=lambda x: int(x[0].split("_")[1])):
                if seq_rps > 0:
                    speedup = metrics["requests_per_second"] / seq_rps

                    if speedup > 1.1:
                        lines.append(f"- {name}: {speedup:.2f}x speedup over sequential")
                    elif speedup < 0.9:
                        lines.append(
                            f"- {name}: **contention detected** ({speedup:.2f}x of sequential)"
                        )
                    else:
                        lines.append(f"- {name}: no significant speedup (mocked backends)")

        # Overall scaling analysis
        concurrent = {k: v for k, v in self.throughput_results.items() if k != "sequential"}
        if len(concurrent) >= 2:
            sorted_concurrent = sorted(
                concurrent.items(),
                key=lambda x: x[1]["requests_per_second"],
                reverse=True,
            )
            best = sorted_concurrent[0]
            worst = sorted_concurrent[-1]

            if worst[1]["requests_per_second"] > 0:
                spread = best[1]["requests_per_second"] / worst[1]["requests_per_second"]
                if spread > 1.2:
                    lines.append(
                        f"- Best concurrency level: {best[0]} "
                        f"({spread:.1f}x better than {worst[0]})"
                    )
                else:
                    lines.append("- Throughput is relatively stable across concurrency levels")

        lines.extend(["", ""])
        return lines
