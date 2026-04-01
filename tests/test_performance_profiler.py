"""
Tests for performance_profiler.py
"""
import time
import pytest
from performance_profiler import PerformanceProfiler, ComponentStats


@pytest.fixture
def profiler() -> PerformanceProfiler:
    return PerformanceProfiler()


class TestPerformanceProfiler:
    def test_context_manager_records_timing(self, profiler):
        with profiler.time("component_a", "op1"):
            time.sleep(0.01)
        stats = profiler.get_stats()
        assert "component_a" in stats
        assert stats["component_a"]["total_calls"] == 1
        assert stats["component_a"]["avg_ms"] > 0

    def test_multiple_calls_aggregate(self, profiler):
        for _ in range(5):
            with profiler.time("comp", "op"):
                time.sleep(0.001)
        stats = profiler.get_stats()
        assert stats["comp"]["total_calls"] == 5

    def test_get_stats_has_required_keys(self, profiler):
        with profiler.time("x", "y"):
            pass
        stats = profiler.get_stats()
        comp_stats = stats["x"]
        assert "avg_ms" in comp_stats
        assert "p95_ms" in comp_stats
        assert "total_calls" in comp_stats

    def test_generate_report_returns_string(self, profiler):
        with profiler.time("comp", "op"):
            time.sleep(0.001)
        report = profiler.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0
        assert "comp" in report

    def test_record_directly(self, profiler):
        profiler.record("direct", "op", 42.5)
        stats = profiler.get_stats()
        assert "direct" in stats
        assert stats["direct"]["avg_ms"] == pytest.approx(42.5, abs=0.1)

    def test_get_stats_with_component_filter(self, profiler):
        with profiler.time("alpha", "op"):
            pass
        with profiler.time("beta", "op"):
            pass
        # Filtering by component returns that component's stats directly
        stats = profiler.get_stats(component="alpha")
        assert "avg_ms" in stats
        assert "total_calls" in stats
        assert stats["component"] == "alpha"
        # Beta should not appear
        assert stats.get("beta") is None

    def test_nested_timing(self, profiler):
        with profiler.time("outer", "op"):
            with profiler.time("inner", "op"):
                time.sleep(0.001)
        stats = profiler.get_stats()
        assert "outer" in stats
        assert "inner" in stats

    def test_p95_computed(self, profiler):
        for i in range(20):
            profiler.record("comp", "op", float(i))
        stats = profiler.get_stats()
        assert stats["comp"]["p95_ms"] >= stats["comp"]["avg_ms"] - 1


class TestComponentStats:
    def test_avg_time_zero_calls(self):
        stats = ComponentStats(component="x")
        assert stats.avg_time_ms == 0.0

    def test_to_dict_keys(self):
        stats = ComponentStats(component="x")
        stats.total_calls = 1
        stats.total_time_ms = 100.0
        stats.samples = [100.0]
        stats.max_time_ms = 100.0
        stats.min_time_ms = 100.0
        d = stats.to_dict()
        assert "avg_ms" in d
        assert "p95_ms" in d
        assert "total_calls" in d
        assert d["total_calls"] == 1
