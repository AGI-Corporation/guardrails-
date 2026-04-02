"""
🚦 Rate Limiter
==============
Token-bucket and sliding-window rate limiters for protecting the Guardrails
API and evaluation endpoints against flooding, DoS, and abuse.

Two algorithms are provided:

    TokenBucketLimiter  — burst-tolerant; refills at a steady rate.
    SlidingWindowLimiter — strict per-period request count; no burst credit.

Both are thread-safe (in-process) and keyed by an arbitrary string (e.g.
``user_id``, ``session_id``, ``ip_address``) so they can be composed.

A convenience ``RateLimitMiddleware`` wraps a callable guardrail pipeline
and raises ``RateLimitExceeded`` when a client exceeds their quota.

Public surface
--------------
    RateLimitExceeded       — exception raised when limit is hit
    RateLimitResult         — result dataclass returned by limiters
    TokenBucketLimiter      — burst-tolerant limiter
    SlidingWindowLimiter    — strict sliding-window limiter
    RateLimitMiddleware     — wraps any callable; raises on limit exceeded
    CompositeRateLimiter    — AND-composition of multiple limiters

Usage
-----
    from rate_limiter import TokenBucketLimiter, RateLimitMiddleware

    limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)

    # Direct check
    result = limiter.check("user-alice")
    if not result.allowed:
        print(f"Rate limited. Retry after {result.retry_after_s:.1f}s")

    # Middleware wrapping guardrail evaluation
    from guardrail_framework import GuardrailEngine, create_default_guardrails
    engine = GuardrailEngine()
    for r in create_default_guardrails():
        engine.add_rule(r)

    middleware = RateLimitMiddleware(
        fn=engine.evaluate,
        limiter=limiter,
        key_fn=lambda text: "global",
    )
    result = middleware("hello world")  # raises RateLimitExceeded if over limit
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional


# ── Exception ─────────────────────────────────────────────────────────────────

class RateLimitExceeded(Exception):
    """Raised by ``RateLimitMiddleware`` when a client exceeds their quota."""

    def __init__(self, key: str, retry_after_s: float = 0.0) -> None:
        self.key = key
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Rate limit exceeded for key '{key}'. "
            f"Retry after {retry_after_s:.2f}s."
        )


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RateLimitResult:
    """Result of a single rate-limit check."""
    allowed: bool
    key: str
    remaining: float        # tokens (bucket) or requests (window) remaining
    reset_after_s: float    # seconds until the bucket/window fully resets
    retry_after_s: float    # seconds to wait before next allowed request (0 if allowed)
    algorithm: str          # "token_bucket" | "sliding_window"
    limit: float            # configured capacity / max_requests


# ── Token-Bucket Limiter ──────────────────────────────────────────────────────

class _BucketState:
    """Per-key mutable state for the token-bucket algorithm."""
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens: float = capacity
        self.last_refill: float = time.monotonic()


class TokenBucketLimiter:
    """
    Token-bucket rate limiter.

    Each key starts with ``capacity`` tokens.  Tokens refill at
    ``refill_rate`` tokens per second up to ``capacity``.  Each call
    to ``check()`` consumes ``cost`` tokens.

    Parameters
    ----------
    capacity:
        Maximum token count (also the initial fill level).
    refill_rate:
        Tokens added per second.
    cost:
        Tokens consumed by each request (default 1.0).
    """

    def __init__(
        self,
        capacity: float = 10.0,
        refill_rate: float = 1.0,
        cost: float = 1.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")
        if cost <= 0:
            raise ValueError("cost must be > 0")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self.cost = cost
        self._buckets: Dict[str, _BucketState] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self, key: str) -> RateLimitResult:
        """
        Check (and consume) quota for *key*.

        Thread-safe.  Does NOT raise — check ``result.allowed``.
        """
        with self._lock:
            state = self._buckets.setdefault(key, _BucketState(self.capacity))
            now = time.monotonic()

            # Refill
            elapsed = now - state.last_refill
            state.tokens = min(self.capacity, state.tokens + elapsed * self.refill_rate)
            state.last_refill = now

            allowed = state.tokens >= self.cost
            if allowed:
                state.tokens -= self.cost

            remaining = max(0.0, state.tokens)
            # Time until tokens replenish enough for one more request
            deficit = max(0.0, self.cost - state.tokens if not allowed else 0.0)
            retry_after = deficit / self.refill_rate if deficit > 0 else 0.0
            reset_after = (self.capacity - remaining) / self.refill_rate

            return RateLimitResult(
                allowed=allowed,
                key=key,
                remaining=remaining,
                reset_after_s=reset_after,
                retry_after_s=retry_after,
                algorithm="token_bucket",
                limit=self.capacity,
            )

    def reset(self, key: str) -> None:
        """Reset the bucket for *key* to full capacity."""
        with self._lock:
            self._buckets.pop(key, None)

    def reset_all(self) -> None:
        """Reset all tracked keys."""
        with self._lock:
            self._buckets.clear()

    def stats(self) -> Dict[str, Dict]:
        """Return current bucket state for all tracked keys (for monitoring)."""
        with self._lock:
            now = time.monotonic()
            out: Dict[str, Dict] = {}
            for k, s in self._buckets.items():
                elapsed = now - s.last_refill
                current = min(self.capacity, s.tokens + elapsed * self.refill_rate)
                out[k] = {
                    "tokens": round(current, 4),
                    "capacity": self.capacity,
                    "utilisation": round(1.0 - current / self.capacity, 4),
                }
            return out


# ── Sliding-Window Limiter ────────────────────────────────────────────────────

class _WindowState:
    """Per-key mutable state for the sliding-window algorithm."""
    __slots__ = ("timestamps",)

    def __init__(self) -> None:
        self.timestamps: Deque[float] = deque()


class SlidingWindowLimiter:
    """
    Sliding-window rate limiter.

    Tracks exact request timestamps for each key.  Allows at most
    ``max_requests`` requests within any rolling ``window_s``-second window.
    No burst credit — the window is strict.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed within the window.
    window_s:
        Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 60, window_s: float = 60.0) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")

        self.max_requests = max_requests
        self.window_s = window_s
        self._windows: Dict[str, _WindowState] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self, key: str) -> RateLimitResult:
        """
        Check (and record) a request for *key*.

        Thread-safe.  Does NOT raise — check ``result.allowed``.
        """
        with self._lock:
            state = self._windows.setdefault(key, _WindowState())
            now = time.monotonic()
            cutoff = now - self.window_s

            # Evict expired timestamps
            while state.timestamps and state.timestamps[0] <= cutoff:
                state.timestamps.popleft()

            count = len(state.timestamps)
            allowed = count < self.max_requests

            if allowed:
                state.timestamps.append(now)
                count += 1

            remaining = max(0, self.max_requests - count)
            # Oldest timestamp in window tells us when a slot frees up
            if not allowed and state.timestamps:
                retry_after = self.window_s - (now - state.timestamps[0])
            else:
                retry_after = 0.0

            reset_after = (
                self.window_s - (now - state.timestamps[0])
                if state.timestamps else 0.0
            )

            return RateLimitResult(
                allowed=allowed,
                key=key,
                remaining=float(remaining),
                reset_after_s=max(0.0, reset_after),
                retry_after_s=max(0.0, retry_after),
                algorithm="sliding_window",
                limit=float(self.max_requests),
            )

    def reset(self, key: str) -> None:
        """Clear all tracked timestamps for *key*."""
        with self._lock:
            self._windows.pop(key, None)

    def reset_all(self) -> None:
        """Clear all tracked keys."""
        with self._lock:
            self._windows.clear()

    def stats(self) -> Dict[str, Dict]:
        """Return current window state for monitoring."""
        with self._lock:
            now = time.monotonic()
            out: Dict[str, Dict] = {}
            for k, s in self._windows.items():
                cutoff = now - self.window_s
                active = sum(1 for ts in s.timestamps if ts > cutoff)
                out[k] = {
                    "active_requests": active,
                    "max_requests": self.max_requests,
                    "utilisation": round(active / self.max_requests, 4),
                }
            return out


# ── Composite Limiter ─────────────────────────────────────────────────────────

class CompositeRateLimiter:
    """
    AND-composition of multiple limiters.

    A request is only allowed when **all** constituent limiters allow it.
    If any limiter denies the request, the most restrictive result is returned.

    Useful for combining per-user and global rate limits:

        composite = CompositeRateLimiter([
            TokenBucketLimiter(capacity=5, refill_rate=0.5),   # per-user burst
            SlidingWindowLimiter(max_requests=100, window_s=60), # global 100/min
        ])
    """

    def __init__(self, limiters: List) -> None:
        if not limiters:
            raise ValueError("At least one limiter is required")
        self.limiters = limiters

    def check(self, key: str) -> RateLimitResult:
        results = [lim.check(key) for lim in self.limiters]
        # Return the most restrictive result (first DENY, else first ALLOW)
        for r in results:
            if not r.allowed:
                return r
        return results[0]

    def reset(self, key: str) -> None:
        for lim in self.limiters:
            lim.reset(key)

    def reset_all(self) -> None:
        for lim in self.limiters:
            lim.reset_all()


# ── Middleware ────────────────────────────────────────────────────────────────

class RateLimitMiddleware:
    """
    Wraps any callable with rate-limit enforcement.

    Parameters
    ----------
    fn:
        The callable to wrap (e.g. ``engine.evaluate``).
    limiter:
        A ``TokenBucketLimiter``, ``SlidingWindowLimiter``, or
        ``CompositeRateLimiter`` instance.
    key_fn:
        Callable that extracts a rate-limit key from the first positional
        argument of ``fn``.  Defaults to a constant ``"global"`` key.
    on_limit:
        What to do when rate-limited: ``"raise"`` (default) raises
        ``RateLimitExceeded``; ``"return_none"`` returns ``None``; or pass a
        zero-argument callable to return a custom value.
    """

    def __init__(
        self,
        fn: Callable,
        limiter,
        key_fn: Optional[Callable[[Any], str]] = None,
        on_limit: Any = "raise",
    ) -> None:
        self._fn = fn
        self._limiter = limiter
        self._key_fn: Callable[[Any], str] = key_fn or (lambda _: "global")
        self._on_limit = on_limit
        self.rate_limit_hits: int = 0
        self.total_requests: int = 0

    def __call__(self, *args, **kwargs) -> Any:
        self.total_requests += 1
        key = self._key_fn(args[0] if args else "")
        result = self._limiter.check(key)

        if not result.allowed:
            self.rate_limit_hits += 1
            if self._on_limit == "raise":
                raise RateLimitExceeded(key=key, retry_after_s=result.retry_after_s)
            if self._on_limit == "return_none":
                return None
            if callable(self._on_limit):
                return self._on_limit()
            raise RateLimitExceeded(key=key, retry_after_s=result.retry_after_s)

        return self._fn(*args, **kwargs)

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.rate_limit_hits / self.total_requests

    def stats(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "rate_limit_hits": self.rate_limit_hits,
            "hit_rate": round(self.hit_rate, 4),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_default_limiter() -> TokenBucketLimiter:
    """Return a sensible default limiter: 60 requests / minute burst-tolerant."""
    return TokenBucketLimiter(capacity=60.0, refill_rate=1.0)


def create_strict_limiter(max_per_minute: int = 30) -> SlidingWindowLimiter:
    """Return a strict sliding-window limiter with *max_per_minute* requests."""
    return SlidingWindowLimiter(max_requests=max_per_minute, window_s=60.0)
