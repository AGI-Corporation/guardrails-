"""
Tests for performance_profiler.py
Covers: PerformanceProfiler record/get_stats/get_bottlenecks/generate_report/reset,
        ComponentStats properties (avg/p50/p95/success_rate/to_dict),
        _TimingContext context manager, timed() decorator.
"""

import time
import pytest

from performance_profiler import (
    ComponentStats,
    ExecutionMetric,
    PerformanceProfiler,
    _TimingContext,
)


# ---------------------------------------------------------------------------
# ComponentStats properties
# ---------------------------------------------------------------------------

class TestComponentStats:
    def test_avg_time_ms_no_calls(self):
        stats = ComponentStats(component="test")
        assert stats.avg_time_ms == 0.0

    def test_avg_time_ms_with_calls(self):
        stats = ComponentStats(component="test", total_calls=2, total_time_ms=100.0)
        assert stats.avg_time_ms == 50.0

    def test_p50_ms_empty(self):
        stats = ComponentStats(component="test")
        assert stats.p50_ms == 0.0

    def test_p50_ms_single_sample(self):
        stats = ComponentStats(component="test", samples=[42.0])
        assert stats.p50_ms == 42.0

    def test_p50_ms_multiple_samples(self):
        stats = ComponentStats(component="test", samples=[10.0, 20.0, 30.0])
        assert stats.p50_ms == 20.0

    def test_p95_ms_empty(self):
        stats = ComponentStats(component="test")
        assert stats.p95_ms == 0.0

    def test_p95_ms_single_sample(self):
        stats = ComponentStats(component="test", samples=[5.0], max_time_ms=5.0)
        assert stats.p95_ms == 5.0

    def test_p95_ms_multiple_samples(self):
        stats = ComponentStats(component="test", samples=list(range(1, 101)))
        # p95 of 1-100 should be around 95-100
        assert stats.p95_ms >= 95

    def test_success_rate_no_calls(self):
        stats = ComponentStats(component="test")
        assert stats.success_rate == 0.0

    def test_success_rate_full(self):
        stats = ComponentStats(component="test", total_calls=4, success_count=4)
        assert stats.success_rate == 100.0

    def test_success_rate_partial(self):
        stats = ComponentStats(component="test", total_calls=4, success_count=2)
        assert stats.success_rate == 50.0

    def test_to_dict_keys(self):
        stats = ComponentStats(component="test", total_calls=1, total_time_ms=10.0,
                               min_time_ms=10.0, max_time_ms=10.0,
                               success_count=1, samples=[10.0])
        d = stats.to_dict()
        for key in ("component", "total_calls", "avg_ms", "min_ms",
                    "max_ms", "p50_ms", "p95_ms", "success_rate_pct"):
            assert key in d

    def test_to_dict_no_calls_min_zero(self):
        stats = ComponentStats(component="test")
        d = stats.to_dict()
        # min should be reported as 0, not inf
        assert d["min_ms"] == 0


# ---------------------------------------------------------------------------
# PerformanceProfiler — record()
# ---------------------------------------------------------------------------

class TestPerformanceProfilerRecord:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_record_creates_component_stats(self):
        self.profiler.record("engine", "evaluate", 5.0)
        assert "engine" in self.profiler._stats

    def test_record_increments_total_calls(self):
        self.profiler.record("engine", "evaluate", 5.0)
        self.profiler.record("engine", "evaluate", 10.0)
        assert self.profiler._stats["engine"].total_calls == 2

    def test_record_accumulates_time(self):
        self.profiler.record("engine", "evaluate", 5.0)
        self.profiler.record("engine", "evaluate", 15.0)
        assert self.profiler._stats["engine"].total_time_ms == 20.0

    def test_record_updates_min_max(self):
        self.profiler.record("engine", "evaluate", 5.0)
        self.profiler.record("engine", "evaluate", 15.0)
        stats = self.profiler._stats["engine"]
        assert stats.min_time_ms == 5.0
        assert stats.max_time_ms == 15.0

    def test_record_success_counts(self):
        self.profiler.record("engine", "evaluate", 1.0, success=True)
        self.profiler.record("engine", "evaluate", 1.0, success=False)
        stats = self.profiler._stats["engine"]
        assert stats.success_count == 1
        assert stats.failure_count == 1

    def test_record_multiple_components(self):
        self.profiler.record("engine", "eval", 1.0)
        self.profiler.record("transformer", "apply", 2.0)
        assert "engine" in self.profiler._stats
        assert "transformer" in self.profiler._stats

    def test_max_samples_enforced_for_metrics(self):
        profiler = PerformanceProfiler(max_samples=5)
        for i in range(10):
            profiler.record("c", "op", float(i))
        assert len(profiler._metrics) <= 5

    def test_max_samples_enforced_for_component_samples(self):
        profiler = PerformanceProfiler(max_samples=3)
        for i in range(10):
            profiler.record("c", "op", float(i))
        assert len(profiler._stats["c"].samples) <= 3


# ---------------------------------------------------------------------------
# PerformanceProfiler — get_stats()
# ---------------------------------------------------------------------------

class TestPerformanceProfilerGetStats:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_get_stats_empty(self):
        assert self.profiler.get_stats() == {}

    def test_get_stats_all_components(self):
        self.profiler.record("a", "op", 1.0)
        self.profiler.record("b", "op", 2.0)
        stats = self.profiler.get_stats()
        assert "a" in stats
        assert "b" in stats

    def test_get_stats_single_component(self):
        self.profiler.record("a", "op", 5.0)
        self.profiler.record("b", "op", 10.0)
        stats = self.profiler.get_stats("a")
        assert "component" in stats
        assert stats["component"] == "a"

    def test_get_stats_unknown_component(self):
        result = self.profiler.get_stats("nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# PerformanceProfiler — get_bottlenecks()
# ---------------------------------------------------------------------------

class TestPerformanceProfilerBottlenecks:
    def test_bottlenecks_sorted_by_avg_desc(self):
        profiler = PerformanceProfiler()
        profiler.record("fast", "op", 1.0)
        profiler.record("medium", "op", 5.0)
        profiler.record("slow", "op", 20.0)
        bottlenecks = profiler.get_bottlenecks(top_n=3)
        assert bottlenecks[0]["component"] == "slow"
        assert bottlenecks[1]["component"] == "medium"
        assert bottlenecks[2]["component"] == "fast"

    def test_bottlenecks_top_n_limit(self):
        profiler = PerformanceProfiler()
        for i in range(10):
            profiler.record(f"c{i}", "op", float(i))
        bottlenecks = profiler.get_bottlenecks(top_n=3)
        assert len(bottlenecks) == 3

    def test_bottlenecks_empty(self):
        profiler = PerformanceProfiler()
        assert profiler.get_bottlenecks() == []


# ---------------------------------------------------------------------------
# PerformanceProfiler — generate_report()
# ---------------------------------------------------------------------------

class TestPerformanceProfilerReport:
    def test_report_contains_header(self):
        profiler = PerformanceProfiler()
        report = profiler.generate_report()
        assert "Performance Profile Report" in report

    def test_report_lists_component(self):
        profiler = PerformanceProfiler()
        profiler.record("my_component", "evaluate", 15.0)
        report = profiler.generate_report()
        assert "my_component" in report

    def test_report_with_multiple_components(self):
        profiler = PerformanceProfiler()
        profiler.record("alpha", "op", 5.0)
        profiler.record("beta", "op", 10.0)
        report = profiler.generate_report()
        assert "alpha" in report
        assert "beta" in report

    def test_report_bottlenecks_section(self):
        profiler = PerformanceProfiler()
        profiler.record("slow", "op", 100.0)
        profiler.record("fast", "op", 1.0)
        report = profiler.generate_report()
        assert "Bottlenecks" in report


# ---------------------------------------------------------------------------
# PerformanceProfiler — reset()
# ---------------------------------------------------------------------------

class TestPerformanceProfilerReset:
    def test_reset_clears_metrics(self):
        profiler = PerformanceProfiler()
        profiler.record("a", "op", 1.0)
        profiler.reset()
        assert profiler._metrics == []

    def test_reset_clears_stats(self):
        profiler = PerformanceProfiler()
        profiler.record("a", "op", 1.0)
        profiler.reset()
        assert profiler._stats == {}

    def test_reset_allows_new_records(self):
        profiler = PerformanceProfiler()
        profiler.record("a", "op", 1.0)
        profiler.reset()
        profiler.record("b", "op", 2.0)
        assert "b" in profiler._stats


# ---------------------------------------------------------------------------
# _TimingContext (context manager via profiler.time())
# ---------------------------------------------------------------------------

class TestTimingContext:
    def test_context_manager_records_metric(self):
        profiler = PerformanceProfiler()
        with profiler.time("engine", "eval"):
            pass
        assert "engine" in profiler._stats

    def test_context_manager_duration_positive(self):
        profiler = PerformanceProfiler()
        with profiler.time("engine", "eval"):
            time.sleep(0.001)
        assert profiler._stats["engine"].total_time_ms > 0

    def test_context_manager_records_failure_on_exception(self):
        profiler = PerformanceProfiler()
        try:
            with profiler.time("engine", "eval"):
                raise ValueError("test error")
        except ValueError:
            pass
        assert profiler._stats["engine"].failure_count == 1

    def test_context_manager_does_not_suppress_exception(self):
        profiler = PerformanceProfiler()
        with pytest.raises(RuntimeError):
            with profiler.time("engine", "eval"):
                raise RuntimeError("should propagate")


# ---------------------------------------------------------------------------
# timed() decorator
# ---------------------------------------------------------------------------

class TestTimedDecorator:
    def test_decorator_records_metric(self):
        profiler = PerformanceProfiler()

        @profiler.timed("my_func", "run")
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        assert "my_func" in profiler._stats

    def test_decorator_records_failure(self):
        profiler = PerformanceProfiler()

        @profiler.timed("broken_func", "run")
        def broken_func():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            broken_func()
        assert profiler._stats["broken_func"].failure_count == 1

    def test_decorator_preserves_function_name(self):
        profiler = PerformanceProfiler()

        @profiler.timed("c", "op")
        def original_name():
            pass

        assert original_name.__name__ == "original_name"
