"""
Performance Profiler
Measures and optimizes guardrail execution time.
"""

import time
import functools
import statistics
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ExecutionMetric:
    timestamp: str
    component: str
    operation: str
    duration_ms: float
    success: bool
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComponentStats:
    component: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    samples: List[float] = field(default_factory=list)

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95_ms(self) -> float:
        if len(self.samples) < 2:
            return self.max_time_ms
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_calls * 100 if self.total_calls > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "total_calls": self.total_calls,
            "avg_ms": round(self.avg_time_ms, 3),
            "min_ms": round(self.min_time_ms if self.min_time_ms != float('inf') else 0, 3),
            "max_ms": round(self.max_time_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "success_rate_pct": round(self.success_rate, 1),
        }


class PerformanceProfiler:
    """Measures and tracks execution time for guardrail components"""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._metrics: List[ExecutionMetric] = []
        self._stats: Dict[str, ComponentStats] = {}

    def record(self, component: str, operation: str, duration_ms: float, success: bool = True):
        metric = ExecutionMetric(
            timestamp=datetime.now().isoformat(),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
        )
        self._metrics.append(metric)
        if len(self._metrics) > self.max_samples:
            self._metrics.pop(0)

        # Update rolling stats
        if component not in self._stats:
            self._stats[component] = ComponentStats(component=component)

        stats = self._stats[component]
        stats.total_calls += 1
        stats.total_time_ms += duration_ms
        stats.min_time_ms = min(stats.min_time_ms, duration_ms)
        stats.max_time_ms = max(stats.max_time_ms, duration_ms)
        stats.samples.append(duration_ms)
        if len(stats.samples) > self.max_samples:
            stats.samples.pop(0)
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

    def time(self, component: str, operation: str = "evaluate"):
        """Context manager for timing a block of code"""
        return _TimingContext(self, component, operation)

    def timed(self, component: str, operation: str = "evaluate"):
        """Decorator for timing a function"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                success = True
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    success = False
                    raise
                finally:
                    duration = (time.perf_counter() - start) * 1000
                    self.record(component, operation, duration, success)
            return wrapper
        return decorator

    def get_stats(self, component: Optional[str] = None) -> Dict:
        if component:
            stats = self._stats.get(component)
            return stats.to_dict() if stats else {}
        return {k: v.to_dict() for k, v in self._stats.items()}

    def get_bottlenecks(self, top_n: int = 5) -> List[Dict]:
        """Return the slowest components by average time"""
        all_stats = [v.to_dict() for v in self._stats.values()]
        return sorted(all_stats, key=lambda x: x["avg_ms"], reverse=True)[:top_n]

    def generate_report(self) -> str:
        lines = [
            "# Performance Profile Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total components tracked: {len(self._stats)}",
            "",
            "## Component Statistics",
            "",
            "| Component | Calls | Avg ms | P95 ms | Max ms | Success% |",
            "|-----------|-------|--------|--------|--------|----------|",
        ]
        for stats in sorted(self._stats.values(), key=lambda s: s.avg_time_ms, reverse=True):
            d = stats.to_dict()
            lines.append(
                f"| {d['component']} | {d['total_calls']} | {d['avg_ms']} | "
                f"{d['p95_ms']} | {d['max_ms']} | {d['success_rate_pct']}% |"
            )

        bottlenecks = self.get_bottlenecks(3)
        if bottlenecks:
            lines.extend([
                "",
                "## Top Bottlenecks",
                "",
            ])
            for b in bottlenecks:
                lines.append(f"- **{b['component']}**: avg {b['avg_ms']}ms, p95 {b['p95_ms']}ms")

        return "\n".join(lines)

    def reset(self):
        self._metrics.clear()
        self._stats.clear()


class _TimingContext:
    def __init__(self, profiler: PerformanceProfiler, component: str, operation: str):
        self.profiler = profiler
        self.component = component
        self.operation = operation
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.perf_counter() - self._start) * 1000
        self.profiler.record(self.component, self.operation, duration, exc_type is None)
        return False  # Don't suppress exceptions


# Global profiler instance
profiler = PerformanceProfiler()
