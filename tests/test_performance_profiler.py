"""
Tests for performance_profiler.py
Covers: PerformanceProfiler (time context manager, get_stats, reset),
        global profiler singleton.
"""

import time
import pytest

from performance_profiler import PerformanceProfiler, profiler


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_profiler():
    p = PerformanceProfiler()
    return p


# ── time context manager ──────────────────────────────────────────────────

class TestTimingContext:

    def test_records_execution(self, fresh_profiler):
        with fresh_profiler.time("svc", "op"):
            time.sleep(0.01)
        stats = fresh_profiler.get_stats("svc")
        assert stats is not None and stats != {}
        assert stats["total_calls"] == 1

    def test_latency_is_positive(self, fresh_profiler):
        with fresh_profiler.time("svc", "op"):
            time.sleep(0.01)
        stats = fresh_profiler.get_stats("svc")
        assert stats["avg_ms"] > 0

    def test_multiple_calls_accumulate(self, fresh_profiler):
        for _ in range(3):
            with fresh_profiler.time("svc", "op"):
                time.sleep(0.001)
        stats = fresh_profiler.get_stats("svc")
        assert stats["total_calls"] == 3

    def test_exception_still_recorded(self, fresh_profiler):
        """Timing should be recorded even if the wrapped code raises."""
        with pytest.raises(ValueError):
            with fresh_profiler.time("svc", "failing_op"):
                raise ValueError("boom")
        stats = fresh_profiler.get_stats("svc")
        assert stats["total_calls"] == 1


# ── get_stats ─────────────────────────────────────────────────────────────

class TestGetStats:

    def test_unknown_service_returns_empty_dict(self, fresh_profiler):
        result = fresh_profiler.get_stats("nonexistent_service")
        assert result == {} or result is None or result.get("total_calls", 0) == 0

    def test_stats_contain_expected_keys(self, fresh_profiler):
        with fresh_profiler.time("api", "evaluate"):
            pass
        stats = fresh_profiler.get_stats("api")
        assert stats is not None and stats != {}
        assert "total_calls" in stats
        assert "avg_ms" in stats


# ── global profiler singleton ─────────────────────────────────────────────

class TestGlobalProfiler:

    def test_global_profiler_is_profiler_instance(self):
        assert isinstance(profiler, PerformanceProfiler)

    def test_global_profiler_can_record(self):
        with profiler.time("global_test", "op"):
            pass
        stats = profiler.get_stats("global_test")
        assert stats is not None and stats != {}
        assert stats.get("total_calls", 0) >= 1
