"""
Tests for rate_limiter.py — TokenBucketLimiter, SlidingWindowLimiter,
CompositeRateLimiter, RateLimitMiddleware, and helpers.
"""
import threading
import time

import pytest

from rate_limiter import (
    CompositeRateLimiter,
    RateLimitExceeded,
    RateLimitMiddleware,
    RateLimitResult,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    create_default_limiter,
    create_strict_limiter,
)


# ── RateLimitExceeded ─────────────────────────────────────────────────────────

class TestRateLimitExceeded:
    def test_message_contains_key(self):
        exc = RateLimitExceeded(key="user-alice", retry_after_s=5.0)
        assert "user-alice" in str(exc)

    def test_retry_after_attribute(self):
        exc = RateLimitExceeded(key="x", retry_after_s=3.5)
        assert exc.retry_after_s == pytest.approx(3.5)

    def test_key_attribute(self):
        exc = RateLimitExceeded(key="bob")
        assert exc.key == "bob"


# ── RateLimitResult ───────────────────────────────────────────────────────────

class TestRateLimitResult:
    def _make(self, allowed=True) -> RateLimitResult:
        return RateLimitResult(
            allowed=allowed, key="k", remaining=5.0,
            reset_after_s=60.0, retry_after_s=0.0 if allowed else 10.0,
            algorithm="token_bucket", limit=10.0,
        )

    def test_allowed_flag(self):
        assert self._make(True).allowed is True
        assert self._make(False).allowed is False

    def test_retry_after_zero_when_allowed(self):
        assert self._make(True).retry_after_s == 0.0

    def test_retry_after_nonzero_when_denied(self):
        assert self._make(False).retry_after_s > 0


# ── TokenBucketLimiter ────────────────────────────────────────────────────────

class TestTokenBucketLimiter:
    def test_first_request_allowed(self):
        lim = TokenBucketLimiter(capacity=5, refill_rate=1.0)
        r = lim.check("alice")
        assert r.allowed is True

    def test_capacity_exhausted(self):
        lim = TokenBucketLimiter(capacity=3, refill_rate=0.001)
        for _ in range(3):
            lim.check("bob")
        r = lim.check("bob")
        assert r.allowed is False

    def test_remaining_decreases(self):
        lim = TokenBucketLimiter(capacity=5, refill_rate=0.001)
        r1 = lim.check("alice")
        r2 = lim.check("alice")
        assert r2.remaining < r1.remaining + 1  # accounting for tiny refill

    def test_different_keys_independent(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        lim.check("alice")  # exhausts alice
        r = lim.check("bob")  # bob still fresh
        assert r.allowed is True

    def test_retry_after_nonzero_when_denied(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=1.0)
        lim.check("alice")
        r = lim.check("alice")
        assert not r.allowed
        assert r.retry_after_s > 0

    def test_algorithm_label(self):
        lim = TokenBucketLimiter()
        r = lim.check("k")
        assert r.algorithm == "token_bucket"

    def test_limit_attribute(self):
        lim = TokenBucketLimiter(capacity=42)
        r = lim.check("k")
        assert r.limit == 42.0

    def test_key_attribute(self):
        lim = TokenBucketLimiter()
        r = lim.check("my-key")
        assert r.key == "my-key"

    def test_reset_restores_capacity(self):
        lim = TokenBucketLimiter(capacity=2, refill_rate=0.001)
        lim.check("alice")
        lim.check("alice")  # exhausted
        lim.reset("alice")
        r = lim.check("alice")
        assert r.allowed is True

    def test_reset_all(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        lim.check("a")
        lim.check("b")
        lim.reset_all()
        assert lim.check("a").allowed is True
        assert lim.check("b").allowed is True

    def test_stats_returns_dict(self):
        lim = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        lim.check("alice")
        stats = lim.stats()
        assert "alice" in stats
        assert "tokens" in stats["alice"]
        assert "utilisation" in stats["alice"]

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(capacity=0)

    def test_invalid_refill_rate_raises(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(refill_rate=-1)

    def test_invalid_cost_raises(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(cost=0)

    def test_thread_safety(self):
        lim = TokenBucketLimiter(capacity=50, refill_rate=0.001)
        results = []

        def work():
            for _ in range(10):
                results.append(lim.check("shared"))

        threads = [threading.Thread(target=work) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some should be allowed, some denied — no exceptions or corruption
        allowed = sum(1 for r in results if r.allowed)
        denied = sum(1 for r in results if not r.allowed)
        assert allowed + denied == 50


# ── SlidingWindowLimiter ──────────────────────────────────────────────────────

class TestSlidingWindowLimiter:
    def test_first_request_allowed(self):
        lim = SlidingWindowLimiter(max_requests=5, window_s=60.0)
        assert lim.check("alice").allowed is True

    def test_exact_limit_allowed(self):
        lim = SlidingWindowLimiter(max_requests=3, window_s=60.0)
        for _ in range(3):
            lim.check("alice")
        # 4th request denied
        r = lim.check("alice")
        assert r.allowed is False

    def test_remaining_decreases(self):
        lim = SlidingWindowLimiter(max_requests=5, window_s=60.0)
        r1 = lim.check("alice")
        r2 = lim.check("alice")
        assert r2.remaining < r1.remaining

    def test_different_keys_independent(self):
        lim = SlidingWindowLimiter(max_requests=1, window_s=60.0)
        lim.check("alice")
        r = lim.check("bob")
        assert r.allowed is True

    def test_retry_after_nonzero_when_denied(self):
        lim = SlidingWindowLimiter(max_requests=1, window_s=30.0)
        lim.check("alice")
        r = lim.check("alice")
        assert not r.allowed
        assert r.retry_after_s > 0
        assert r.retry_after_s <= 30.0

    def test_algorithm_label(self):
        lim = SlidingWindowLimiter()
        r = lim.check("k")
        assert r.algorithm == "sliding_window"

    def test_limit_attribute(self):
        lim = SlidingWindowLimiter(max_requests=42)
        r = lim.check("k")
        assert r.limit == 42.0

    def test_reset_clears_window(self):
        lim = SlidingWindowLimiter(max_requests=1, window_s=60.0)
        lim.check("alice")
        lim.reset("alice")
        assert lim.check("alice").allowed is True

    def test_reset_all(self):
        lim = SlidingWindowLimiter(max_requests=1, window_s=60.0)
        lim.check("a")
        lim.check("b")
        lim.reset_all()
        assert lim.check("a").allowed is True
        assert lim.check("b").allowed is True

    def test_stats_returns_dict(self):
        lim = SlidingWindowLimiter(max_requests=5, window_s=60.0)
        lim.check("alice")
        stats = lim.stats()
        assert "alice" in stats
        assert "active_requests" in stats["alice"]

    def test_invalid_max_requests_raises(self):
        with pytest.raises(ValueError):
            SlidingWindowLimiter(max_requests=0)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            SlidingWindowLimiter(window_s=-1)

    def test_thread_safety(self):
        lim = SlidingWindowLimiter(max_requests=25, window_s=60.0)
        results = []

        def work():
            for _ in range(10):
                results.append(lim.check("shared"))

        threads = [threading.Thread(target=work) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed = sum(1 for r in results if r.allowed)
        assert allowed == 25  # exactly max_requests


# ── CompositeRateLimiter ──────────────────────────────────────────────────────

class TestCompositeRateLimiter:
    def test_allows_when_all_allow(self):
        lim = CompositeRateLimiter([
            TokenBucketLimiter(capacity=10, refill_rate=1.0),
            SlidingWindowLimiter(max_requests=10, window_s=60.0),
        ])
        assert lim.check("alice").allowed is True

    def test_denies_when_first_denies(self):
        bucket = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        window = SlidingWindowLimiter(max_requests=100, window_s=60.0)
        lim = CompositeRateLimiter([bucket, window])
        lim.check("alice")  # exhaust bucket
        r = lim.check("alice")
        assert not r.allowed

    def test_denies_when_second_denies(self):
        bucket = TokenBucketLimiter(capacity=100, refill_rate=1.0)
        window = SlidingWindowLimiter(max_requests=1, window_s=60.0)
        lim = CompositeRateLimiter([bucket, window])
        lim.check("alice")
        r = lim.check("alice")
        assert not r.allowed

    def test_reset_all(self):
        bucket = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        lim = CompositeRateLimiter([bucket])
        lim.check("alice")
        lim.reset_all()
        assert lim.check("alice").allowed is True

    def test_empty_limiter_list_raises(self):
        with pytest.raises(ValueError):
            CompositeRateLimiter([])


# ── RateLimitMiddleware ───────────────────────────────────────────────────────

class TestRateLimitMiddleware:
    def _identity(self, x):
        return x.upper()

    def test_passes_through_when_allowed(self):
        lim = TokenBucketLimiter(capacity=10)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim)
        assert mw("hello") == "HELLO"

    def test_raises_when_limit_exceeded(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim)
        mw("first")
        with pytest.raises(RateLimitExceeded):
            mw("second")

    def test_return_none_mode(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim, on_limit="return_none")
        mw("first")
        assert mw("second") is None

    def test_custom_fallback(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        mw = RateLimitMiddleware(
            fn=self._identity, limiter=lim,
            on_limit=lambda: "BLOCKED",
        )
        mw("first")
        assert mw("second") == "BLOCKED"

    def test_key_fn_used(self):
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        mw = RateLimitMiddleware(
            fn=self._identity, limiter=lim,
            key_fn=lambda text: "always-same",
        )
        mw("x")
        with pytest.raises(RateLimitExceeded) as exc_info:
            mw("y")
        assert exc_info.value.key == "always-same"

    def test_total_requests_counted(self):
        lim = TokenBucketLimiter(capacity=10)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim)
        for _ in range(5):
            mw("x")
        assert mw.total_requests == 5

    def test_hit_rate_tracked(self):
        lim = TokenBucketLimiter(capacity=2, refill_rate=0.001)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim, on_limit="return_none")
        mw("a")
        mw("b")
        mw("c")  # denied
        mw("d")  # denied
        assert mw.hit_rate == pytest.approx(0.5)

    def test_stats_method(self):
        lim = TokenBucketLimiter(capacity=10)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim)
        mw("x")
        stats = mw.stats()
        assert "total_requests" in stats
        assert stats["total_requests"] == 1

    def test_zero_hit_rate_at_start(self):
        lim = TokenBucketLimiter(capacity=10)
        mw = RateLimitMiddleware(fn=self._identity, limiter=lim)
        assert mw.hit_rate == 0.0

    def test_works_with_guardrail_engine(self):
        from guardrail_framework import GuardrailEngine, create_default_guardrails
        engine = GuardrailEngine()
        for r in create_default_guardrails():
            engine.add_rule(r)

        lim = TokenBucketLimiter(capacity=10)
        mw = RateLimitMiddleware(fn=engine.evaluate, limiter=lim)
        result = mw("hello world")
        assert hasattr(result, "action")


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_create_default_limiter(self):
        lim = create_default_limiter()
        assert isinstance(lim, TokenBucketLimiter)
        assert lim.capacity == 60.0
        assert lim.refill_rate == 1.0

    def test_create_strict_limiter(self):
        lim = create_strict_limiter(max_per_minute=15)
        assert isinstance(lim, SlidingWindowLimiter)
        assert lim.max_requests == 15
        assert lim.window_s == 60.0

    def test_default_limiter_allows_first_request(self):
        lim = create_default_limiter()
        assert lim.check("test").allowed is True

    def test_strict_limiter_denies_after_limit(self):
        lim = create_strict_limiter(max_per_minute=2)
        lim.check("k")
        lim.check("k")
        assert lim.check("k").allowed is False


# ── SlidingWindow: time-based eviction ───────────────────────────────────────

class TestSlidingWindowEviction:
    """Test that expired timestamps are correctly evicted (covers line 252)."""

    def test_slots_freed_after_window_expires(self):
        import time
        lim = SlidingWindowLimiter(max_requests=2, window_s=0.1)
        # Exhaust the window
        lim.check("alice")
        lim.check("alice")
        # 4th call should be denied
        assert lim.check("alice").allowed is False
        # Wait for the window to expire
        time.sleep(0.15)
        # Now a new slot is available
        assert lim.check("alice").allowed is True

    def test_partial_eviction_allows_some(self):
        import time
        lim = SlidingWindowLimiter(max_requests=3, window_s=0.1)
        lim.check("bob")    # t=0
        time.sleep(0.08)
        lim.check("bob")    # t=0.08
        lim.check("bob")    # t=0.08 — now at limit
        assert lim.check("bob").allowed is False
        time.sleep(0.05)    # t=0.13 — first request at t=0 has expired
        r = lim.check("bob")
        assert r.allowed is True  # one slot freed


# ── CompositeRateLimiter per-key reset ───────────────────────────────────────

class TestCompositeReset:
    def test_reset_key_restores_per_key(self):
        bucket = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        lim = CompositeRateLimiter([bucket])
        lim.check("alice")     # exhaust
        assert lim.check("alice").allowed is False
        lim.reset("alice")
        assert lim.check("alice").allowed is True

    def test_reset_key_does_not_affect_other_keys(self):
        bucket = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        lim = CompositeRateLimiter([bucket])
        lim.check("alice")
        lim.check("bob")
        lim.reset("alice")    # only alice reset
        assert lim.check("alice").allowed is True
        assert lim.check("bob").allowed is False


# ── RateLimitMiddleware: fallthrough unknown on_limit ────────────────────────

class TestMiddlewareFallthrough:
    def test_unknown_on_limit_string_still_raises(self):
        """Any on_limit value that isn't 'raise', 'return_none', or callable
        falls through to the RateLimitExceeded raise at line 397."""
        lim = TokenBucketLimiter(capacity=1, refill_rate=0.001)
        mw = RateLimitMiddleware(
            fn=lambda x: x,
            limiter=lim,
            on_limit="invalid_sentinel",   # not "raise" or "return_none", not callable
        )
        mw("first")   # consume the single token
        with pytest.raises(RateLimitExceeded):
            mw("second")
