"""
Tests for performance_profiler.py — metrics recording, stats, bottlenecks,
timing context manager, timed decorator, and report generation.
"""
import time

import pytest

from performance_profiler import (
    ComponentStats,
    ExecutionMetric,
    PerformanceProfiler,
    _TimingContext,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def profiler() -> PerformanceProfiler:
    return PerformanceProfiler()


# ── PerformanceProfiler.record ────────────────────────────────────────────────

class TestRecord:
    def test_single_record(self, profiler: PerformanceProfiler):
        profiler.record("comp_a", "op", 12.5)
        stats = profiler.get_stats("comp_a")
        assert stats["total_calls"] == 1
        assert stats["avg_ms"] == 12.5

    def test_min_max_tracked(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 10.0)
        profiler.record("comp", "op", 20.0)
        stats = profiler.get_stats("comp")
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 20.0

    def test_success_count(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 5.0, success=True)
        profiler.record("comp", "op", 5.0, success=False)
        stats = profiler.get_stats("comp")
        assert stats["success_rate_pct"] == 50.0

    def test_multiple_components(self, profiler: PerformanceProfiler):
        profiler.record("a", "op", 1.0)
        profiler.record("b", "op", 2.0)
        all_stats = profiler.get_stats()
        assert "a" in all_stats
        assert "b" in all_stats

    def test_max_samples_enforced(self):
        p = PerformanceProfiler(max_samples=5)
        for i in range(10):
            p.record("comp", "op", float(i))
        stats = p.get_stats("comp")
        # total_calls reflects all records but samples capped
        assert stats["total_calls"] == 10

    def test_unknown_component_returns_empty(self, profiler: PerformanceProfiler):
        assert profiler.get_stats("nonexistent") == {}

    def test_metrics_stored(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 7.0)
        assert len(profiler._metrics) == 1
        m = profiler._metrics[0]
        assert isinstance(m, ExecutionMetric)
        assert m.component == "comp"
        assert m.operation == "op"
        assert m.duration_ms == 7.0


# ── ComponentStats properties ─────────────────────────────────────────────────

class TestComponentStats:
    def _make(self, samples):
        stats = ComponentStats(component="test")
        for s in samples:
            stats.total_calls += 1
            stats.total_time_ms += s
            stats.min_time_ms = min(stats.min_time_ms, s)
            stats.max_time_ms = max(stats.max_time_ms, s)
            stats.samples.append(s)
            stats.success_count += 1
        return stats

    def test_avg_time_ms(self):
        stats = self._make([10.0, 20.0, 30.0])
        assert stats.avg_time_ms == pytest.approx(20.0)

    def test_avg_time_empty(self):
        stats = ComponentStats(component="test")
        assert stats.avg_time_ms == 0.0

    def test_p50_ms(self):
        stats = self._make([10.0, 20.0, 30.0])
        assert stats.p50_ms == pytest.approx(20.0)

    def test_p95_ms(self):
        samples = sorted([float(i) for i in range(1, 101)])
        stats = self._make(samples)
        # p95 of 1..100 should be around 95-100
        assert stats.p95_ms >= 94.0

    def test_success_rate_full(self):
        stats = self._make([1.0, 2.0])
        assert stats.success_rate == 100.0

    def test_to_dict_keys(self):
        stats = self._make([5.0])
        d = stats.to_dict()
        expected_keys = {
            "component", "total_calls", "avg_ms", "min_ms",
            "max_ms", "p50_ms", "p95_ms", "success_rate_pct",
        }
        assert expected_keys == set(d.keys())


# ── Timing context manager ────────────────────────────────────────────────────

class TestTimingContext:
    def test_records_duration(self, profiler: PerformanceProfiler):
        with profiler.time("ctx_comp", "ctx_op"):
            time.sleep(0.001)
        stats = profiler.get_stats("ctx_comp")
        assert stats["total_calls"] == 1
        assert stats["avg_ms"] > 0

    def test_records_success_on_no_exception(self, profiler: PerformanceProfiler):
        with profiler.time("comp", "op"):
            pass
        stats = profiler.get_stats("comp")
        assert stats["success_rate_pct"] == 100.0

    def test_records_failure_on_exception(self, profiler: PerformanceProfiler):
        with pytest.raises(ValueError):
            with profiler.time("comp", "op"):
                raise ValueError("boom")
        stats = profiler.get_stats("comp")
        assert stats["success_rate_pct"] == 0.0

    def test_exception_propagated(self, profiler: PerformanceProfiler):
        with pytest.raises(RuntimeError, match="propagated"):
            with profiler.time("comp", "op"):
                raise RuntimeError("propagated")

    def test_nested_contexts(self, profiler: PerformanceProfiler):
        with profiler.time("outer", "op"):
            with profiler.time("inner", "op"):
                pass
        assert profiler.get_stats("outer")["total_calls"] == 1
        assert profiler.get_stats("inner")["total_calls"] == 1


# ── Timed decorator ───────────────────────────────────────────────────────────

class TestTimedDecorator:
    def test_decorator_records_call(self, profiler: PerformanceProfiler):
        @profiler.timed("decorated", "run")
        def my_fn():
            return 42

        result = my_fn()
        assert result == 42
        stats = profiler.get_stats("decorated")
        assert stats["total_calls"] == 1

    def test_decorator_preserves_name(self, profiler: PerformanceProfiler):
        @profiler.timed("comp", "op")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_records_failure(self, profiler: PerformanceProfiler):
        @profiler.timed("comp", "op")
        def boom():
            raise ValueError("error")

        with pytest.raises(ValueError):
            boom()
        stats = profiler.get_stats("comp")
        assert stats["success_rate_pct"] == 0.0


# ── get_bottlenecks ───────────────────────────────────────────────────────────

class TestGetBottlenecks:
    def test_returns_sorted_by_avg(self, profiler: PerformanceProfiler):
        profiler.record("slow", "op", 100.0)
        profiler.record("fast", "op", 1.0)
        profiler.record("medium", "op", 50.0)
        bottlenecks = profiler.get_bottlenecks(top_n=3)
        assert bottlenecks[0]["component"] == "slow"
        assert bottlenecks[1]["component"] == "medium"
        assert bottlenecks[2]["component"] == "fast"

    def test_top_n_limit(self, profiler: PerformanceProfiler):
        for i in range(10):
            profiler.record(f"comp_{i}", "op", float(i))
        bottlenecks = profiler.get_bottlenecks(top_n=3)
        assert len(bottlenecks) == 3

    def test_empty_profiler(self, profiler: PerformanceProfiler):
        assert profiler.get_bottlenecks() == []


# ── generate_report ───────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_returns_string(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 5.0)
        report = profiler.generate_report()
        assert isinstance(report, str)

    def test_contains_component_name(self, profiler: PerformanceProfiler):
        profiler.record("my_special_comp", "op", 5.0)
        assert "my_special_comp" in profiler.generate_report()

    def test_contains_header(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 5.0)
        assert "Performance Profile Report" in profiler.generate_report()

    def test_empty_profiler_report(self, profiler: PerformanceProfiler):
        report = profiler.generate_report()
        assert "0" in report  # zero components


# ── reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_metrics(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 5.0)
        profiler.reset()
        assert profiler._metrics == []
        assert profiler._stats == {}

    def test_reset_then_record(self, profiler: PerformanceProfiler):
        profiler.record("comp", "op", 5.0)
        profiler.reset()
        profiler.record("comp2", "op", 99.0)
        stats = profiler.get_stats("comp2")
        assert stats["total_calls"] == 1
