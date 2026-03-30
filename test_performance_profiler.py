"""
Tests for performance_profiler.py
Covers PerformanceProfiler, ComponentStats, ExecutionMetric, _TimingContext,
and the timed decorator.
"""

import time

import pytest

from performance_profiler import (
    ComponentStats,
    ExecutionMetric,
    PerformanceProfiler,
    profiler as global_profiler,
)


# ── ComponentStats ────────────────────────────────────────────────────────────

class TestComponentStats:
    def test_avg_time_no_calls(self):
        stats = ComponentStats(component="test")
        assert stats.avg_time_ms == 0.0

    def test_avg_time_with_calls(self):
        stats = ComponentStats(
            component="test",
            total_calls=2,
            total_time_ms=20.0,
        )
        assert stats.avg_time_ms == pytest.approx(10.0)

    def test_p50_no_samples(self):
        stats = ComponentStats(component="test")
        assert stats.p50_ms == 0.0

    def test_p50_odd_samples(self):
        stats = ComponentStats(component="test", samples=[10.0, 20.0, 30.0])
        assert stats.p50_ms == pytest.approx(20.0)

    def test_p95_single_sample(self):
        stats = ComponentStats(component="test", samples=[5.0])
        # Falls back to max_time_ms when < 2 samples
        assert stats.p95_ms == stats.max_time_ms

    def test_p95_multiple_samples(self):
        stats = ComponentStats(
            component="test",
            samples=list(range(1, 101)),  # 1..100
        )
        assert stats.p95_ms >= 95

    def test_success_rate_no_calls(self):
        stats = ComponentStats(component="test")
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        stats = ComponentStats(
            component="test",
            total_calls=5,
            success_count=5,
        )
        assert stats.success_rate == pytest.approx(100.0)

    def test_success_rate_partial(self):
        stats = ComponentStats(
            component="test",
            total_calls=4,
            success_count=3,
        )
        assert stats.success_rate == pytest.approx(75.0)

    def test_to_dict_keys(self):
        stats = ComponentStats(component="engine", total_calls=1, total_time_ms=5.0,
                               min_time_ms=5.0, max_time_ms=5.0, success_count=1,
                               samples=[5.0])
        d = stats.to_dict()
        for key in ("component", "total_calls", "avg_ms", "min_ms", "max_ms",
                    "p50_ms", "p95_ms", "success_rate_pct"):
            assert key in d


# ── PerformanceProfiler ───────────────────────────────────────────────────────

class TestPerformanceProfilerRecord:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_record_creates_component_stats(self):
        self.profiler.record("engine", "evaluate", 5.0)
        stats = self.profiler.get_stats("engine")
        assert stats["total_calls"] == 1

    def test_record_accumulates(self):
        self.profiler.record("engine", "evaluate", 10.0)
        self.profiler.record("engine", "evaluate", 20.0)
        stats = self.profiler.get_stats("engine")
        assert stats["total_calls"] == 2
        assert stats["avg_ms"] == pytest.approx(15.0)

    def test_record_failure_tracked(self):
        self.profiler.record("engine", "evaluate", 5.0, success=False)
        stats = self.profiler.get_stats("engine")
        assert stats["success_rate_pct"] == pytest.approx(0.0)

    def test_record_mixed_success_failure(self):
        self.profiler.record("engine", "evaluate", 1.0, success=True)
        self.profiler.record("engine", "evaluate", 1.0, success=False)
        stats = self.profiler.get_stats("engine")
        assert stats["success_rate_pct"] == pytest.approx(50.0)

    def test_max_samples_respected(self):
        profiler = PerformanceProfiler(max_samples=5)
        for i in range(10):
            profiler.record("engine", "evaluate", float(i))
        # Internal metric list should not exceed max_samples
        assert len(profiler._metrics) <= 5

    def test_multiple_components(self):
        self.profiler.record("engine", "evaluate", 5.0)
        self.profiler.record("transformer", "transform", 2.0)
        all_stats = self.profiler.get_stats()
        assert "engine" in all_stats
        assert "transformer" in all_stats

    def test_get_stats_unknown_component_returns_empty(self):
        result = self.profiler.get_stats("nonexistent")
        assert result == {}

    def test_min_max_tracked(self):
        self.profiler.record("engine", "evaluate", 3.0)
        self.profiler.record("engine", "evaluate", 7.0)
        stats = self.profiler.get_stats("engine")
        assert stats["min_ms"] == pytest.approx(3.0)
        assert stats["max_ms"] == pytest.approx(7.0)


class TestPerformanceProfilerTimingContext:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_context_manager_records_duration(self):
        with self.profiler.time("engine", "evaluate"):
            time.sleep(0.01)  # 10 ms
        stats = self.profiler.get_stats("engine")
        assert stats["total_calls"] == 1
        assert stats["avg_ms"] >= 5  # At least 5ms

    def test_context_manager_records_failure_on_exception(self):
        with pytest.raises(ValueError):
            with self.profiler.time("engine", "evaluate"):
                raise ValueError("test error")
        stats = self.profiler.get_stats("engine")
        assert stats["total_calls"] == 1
        assert stats["success_rate_pct"] == pytest.approx(0.0)

    def test_context_manager_does_not_suppress_exception(self):
        with pytest.raises(RuntimeError):
            with self.profiler.time("engine", "evaluate"):
                raise RuntimeError("should propagate")


class TestPerformanceProfilerTimedDecorator:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_decorator_records_call(self):
        @self.profiler.timed("engine", "evaluate")
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        stats = self.profiler.get_stats("engine")
        assert stats["total_calls"] == 1

    def test_decorator_records_failure(self):
        @self.profiler.timed("engine", "evaluate")
        def failing_func():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            failing_func()

        stats = self.profiler.get_stats("engine")
        assert stats["total_calls"] == 1
        assert stats["success_rate_pct"] == pytest.approx(0.0)

    def test_decorator_preserves_function_name(self):
        @self.profiler.timed("engine", "evaluate")
        def my_named_func():
            pass

        assert my_named_func.__name__ == "my_named_func"


class TestPerformanceProfilerBottlenecks:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_bottlenecks_sorted_by_avg_desc(self):
        self.profiler.record("fast", "op", 1.0)
        self.profiler.record("slow", "op", 100.0)
        bottlenecks = self.profiler.get_bottlenecks(top_n=5)
        assert bottlenecks[0]["component"] == "slow"
        assert bottlenecks[1]["component"] == "fast"

    def test_bottlenecks_top_n_respected(self):
        for i in range(5):
            self.profiler.record(f"comp{i}", "op", float(i * 10))
        bottlenecks = self.profiler.get_bottlenecks(top_n=2)
        assert len(bottlenecks) == 2

    def test_bottlenecks_empty_profiler(self):
        assert self.profiler.get_bottlenecks() == []


class TestPerformanceProfilerGenerateReport:
    def setup_method(self):
        self.profiler = PerformanceProfiler()

    def test_report_is_string(self):
        report = self.profiler.generate_report()
        assert isinstance(report, str)

    def test_report_contains_header(self):
        report = self.profiler.generate_report()
        assert "Performance Profile Report" in report

    def test_report_with_data_contains_component(self):
        self.profiler.record("my_engine", "evaluate", 5.0)
        report = self.profiler.generate_report()
        assert "my_engine" in report

    def test_report_contains_bottlenecks_section_when_data(self):
        self.profiler.record("slow_component", "op", 999.0)
        report = self.profiler.generate_report()
        assert "Bottleneck" in report or "slow_component" in report


class TestPerformanceProfilerReset:
    def test_reset_clears_metrics_and_stats(self):
        profiler = PerformanceProfiler()
        profiler.record("engine", "evaluate", 5.0)
        profiler.reset()
        assert profiler.get_stats() == {}
        assert profiler._metrics == []


class TestGlobalProfiler:
    def test_global_profiler_is_instance(self):
        assert isinstance(global_profiler, PerformanceProfiler)
