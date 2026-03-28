"""
Tests for performance_profiler.py
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
# ComponentStats
# ---------------------------------------------------------------------------

class TestComponentStats:
    def _make_stats(self, samples) -> ComponentStats:
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
        stats = self._make_stats([10.0, 20.0, 30.0])
        assert abs(stats.avg_time_ms - 20.0) < 0.001

    def test_avg_time_ms_no_calls(self):
        stats = ComponentStats(component="empty")
        assert stats.avg_time_ms == 0.0

    def test_p50_ms(self):
        stats = self._make_stats([10.0, 20.0, 30.0])
        assert stats.p50_ms == 20.0

    def test_p95_ms_with_many_samples(self):
        samples = list(range(1, 101))  # 1..100 ms
        stats = self._make_stats([float(s) for s in samples])
        # p95 should be ~95
        assert 94 <= stats.p95_ms <= 100

    def test_p95_ms_single_sample(self):
        stats = self._make_stats([42.0])
        assert stats.p95_ms == 42.0

    def test_success_rate_100(self):
        stats = self._make_stats([10.0, 20.0])
        assert stats.success_rate == 100.0

    def test_success_rate_partial(self):
        stats = ComponentStats(component="test")
        stats.total_calls = 4
        stats.success_count = 3
        stats.failure_count = 1
        assert stats.success_rate == 75.0

    def test_success_rate_zero_calls(self):
        stats = ComponentStats(component="empty")
        assert stats.success_rate == 0.0

    def test_to_dict_keys(self):
        stats = self._make_stats([10.0])
        d = stats.to_dict()
        for key in ["component", "total_calls", "avg_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "success_rate_pct"]:
            assert key in d

    def test_to_dict_values_rounded(self):
        stats = self._make_stats([10.123456])
        d = stats.to_dict()
        # Should be rounded to 3 decimal places
        assert d["avg_ms"] == round(10.123456, 3)

    def test_min_max_tracked(self):
        stats = self._make_stats([5.0, 15.0, 10.0])
        assert stats.min_time_ms == 5.0
        assert stats.max_time_ms == 15.0


# ---------------------------------------------------------------------------
# PerformanceProfiler — record
# ---------------------------------------------------------------------------

class TestPerformanceProfilerRecord:
    def test_record_creates_component_stats(self):
        profiler = PerformanceProfiler()
        profiler.record("engine", "evaluate", 5.0)
        assert "engine" in profiler._stats

    def test_record_multiple_times(self):
        profiler = PerformanceProfiler()
        for _ in range(3):
            profiler.record("engine", "evaluate", 10.0)
        assert profiler._stats["engine"].total_calls == 3

    def test_record_failure_increments_failure_count(self):
        profiler = PerformanceProfiler()
        profiler.record("comp", "op", 5.0, success=False)
        assert profiler._stats["comp"].failure_count == 1
        assert profiler._stats["comp"].success_count == 0

    def test_record_success_increments_success_count(self):
        profiler = PerformanceProfiler()
        profiler.record("comp", "op", 5.0, success=True)
        assert profiler._stats["comp"].success_count == 1

    def test_max_samples_enforced(self):
        profiler = PerformanceProfiler(max_samples=5)
        for i in range(10):
            profiler.record("comp", "op", float(i))
        # metrics list should not exceed max_samples
        assert len(profiler._metrics) <= 5


# ---------------------------------------------------------------------------
# PerformanceProfiler — get_stats
# ---------------------------------------------------------------------------

class TestPerformanceProfilerGetStats:
    def test_get_stats_all_components(self):
        profiler = PerformanceProfiler()
        profiler.record("comp_a", "op1", 10.0)
        profiler.record("comp_b", "op2", 20.0)
        stats = profiler.get_stats()
        assert "comp_a" in stats
        assert "comp_b" in stats

    def test_get_stats_specific_component(self):
        profiler = PerformanceProfiler()
        profiler.record("my_comp", "op", 15.0)
        stats = profiler.get_stats("my_comp")
        assert stats["component"] == "my_comp"
        assert stats["total_calls"] == 1

    def test_get_stats_nonexistent_component(self):
        profiler = PerformanceProfiler()
        stats = profiler.get_stats("nonexistent")
        assert stats == {}

    def test_get_stats_empty_profiler(self):
        profiler = PerformanceProfiler()
        stats = profiler.get_stats()
        assert stats == {}


# ---------------------------------------------------------------------------
# PerformanceProfiler — get_bottlenecks
# ---------------------------------------------------------------------------

class TestPerformanceProfilerBottlenecks:
    def test_bottlenecks_ordered_by_avg_time(self):
        profiler = PerformanceProfiler()
        profiler.record("fast", "op", 1.0)
        profiler.record("slow", "op", 100.0)
        profiler.record("medium", "op", 50.0)
        bottlenecks = profiler.get_bottlenecks(top_n=2)
        assert bottlenecks[0]["component"] == "slow"
        assert bottlenecks[1]["component"] == "medium"

    def test_bottlenecks_top_n_respected(self):
        profiler = PerformanceProfiler()
        for i in range(5):
            profiler.record(f"comp_{i}", "op", float(i * 10))
        bottlenecks = profiler.get_bottlenecks(top_n=3)
        assert len(bottlenecks) == 3

    def test_bottlenecks_empty_profiler(self):
        profiler = PerformanceProfiler()
        bottlenecks = profiler.get_bottlenecks()
        assert bottlenecks == []


# ---------------------------------------------------------------------------
# PerformanceProfiler — timing context manager
# ---------------------------------------------------------------------------

class TestTimingContext:
    def test_records_duration(self):
        profiler = PerformanceProfiler()
        with profiler.time("test_comp", "test_op"):
            time.sleep(0.005)
        stats = profiler.get_stats("test_comp")
        assert stats["total_calls"] == 1
        assert stats["avg_ms"] > 0

    def test_records_success_on_clean_exit(self):
        profiler = PerformanceProfiler()
        with profiler.time("test_component", "test_operation"):
            pass
        assert profiler._stats["test_component"].success_count == 1
        assert profiler._stats["test_component"].failure_count == 0

    def test_records_failure_on_exception(self):
        profiler = PerformanceProfiler()
        with pytest.raises(ValueError):
            with profiler.time("test_component", "test_operation"):
                raise ValueError("oops")
        assert profiler._stats["test_component"].failure_count == 1
        assert profiler._stats["test_component"].success_count == 0

    def test_exception_propagates(self):
        profiler = PerformanceProfiler()
        with pytest.raises(RuntimeError, match="propagated"):
            with profiler.time("test_component", "test_operation"):
                raise RuntimeError("propagated")


# ---------------------------------------------------------------------------
# PerformanceProfiler — timed decorator
# ---------------------------------------------------------------------------

class TestTimedDecorator:
    def test_decorator_records_call(self):
        profiler = PerformanceProfiler()

        @profiler.timed("my_comp", "run")
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        stats = profiler.get_stats("my_comp")
        assert stats["total_calls"] == 1

    def test_decorator_records_failure(self):
        profiler = PerformanceProfiler()

        @profiler.timed("failing_comp", "run")
        def failing_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            failing_func()
        assert profiler._stats["failing_comp"].failure_count == 1

    def test_decorator_preserves_function_name(self):
        profiler = PerformanceProfiler()

        @profiler.timed("comp", "op")
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"


# ---------------------------------------------------------------------------
# PerformanceProfiler — generate_report
# ---------------------------------------------------------------------------

class TestPerformanceProfilerReport:
    def test_report_contains_component_name(self):
        profiler = PerformanceProfiler()
        profiler.record("engine", "evaluate", 5.0)
        report = profiler.generate_report()
        assert "engine" in report

    def test_report_structure(self):
        profiler = PerformanceProfiler()
        profiler.record("comp", "op", 10.0)
        report = profiler.generate_report()
        assert "Performance Profile Report" in report
        assert "Component Statistics" in report

    def test_empty_profiler_report(self):
        profiler = PerformanceProfiler()
        report = profiler.generate_report()
        assert isinstance(report, str)
        assert "0" in report  # 0 components


# ---------------------------------------------------------------------------
# PerformanceProfiler — reset
# ---------------------------------------------------------------------------

class TestPerformanceProfilerReset:
    def test_reset_clears_metrics_and_stats(self):
        profiler = PerformanceProfiler()
        profiler.record("comp", "op", 10.0)
        profiler.reset()
        assert profiler._metrics == []
        assert profiler._stats == {}

    def test_after_reset_fresh_recording_works(self):
        profiler = PerformanceProfiler()
        profiler.record("comp", "op", 10.0)
        profiler.reset()
        profiler.record("new_comp", "op2", 5.0)
        assert "new_comp" in profiler._stats
        assert "comp" not in profiler._stats
