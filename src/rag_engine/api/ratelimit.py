"""Sliding-window rate limiting keyed on VERIFIED identity (spec #61).

Called from the auth dependency (api/dependencies.get_auth_context), so:
  * authenticated callers are limited per (tenant, user) -- a caller cannot get a fresh
    bucket by changing headers;
  * failed authentications are limited per client IP, which throttles credential guessing
    and fake-token floods (they get 429 instead of unlimited 401s);
  * errors are raised inside request handling, where the app's exception handler turns
    them into a proper 429 (raising from middleware would surface as a 500).

In-memory and per-process: a multi-instance deployment needs a shared store (e.g. Redis).
Memory is bounded: expired entries are purged and the least-recently-used key is evicted
once `max_keys` is reached.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from rag_engine.exceptions import RateLimitExceededError


class SlidingWindowLimiter:
    def __init__(self, window_s: float = 60.0, max_keys: int = 10_000, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = window_s
        self.max_keys = max_keys
        self._clock = clock
        self._windows: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> None:
        """Record one hit for `key`; raise RateLimitExceededError if it would exceed `limit` per window."""
        now = self._clock()
        with self._lock:
            window = self._windows.get(key)
            if window is None:
                window = self._windows[key] = deque()
                while len(self._windows) > self.max_keys:
                    self._windows.popitem(last=False)  # evict least recently used
            else:
                self._windows.move_to_end(key)
            while window and now - window[0] >= self.window_s:
                window.popleft()
            if len(window) >= limit:
                retry_after = max(1, int(self.window_s - (now - window[0])) + 1)
                raise RateLimitExceededError(f"rate limit of {limit}/min exceeded", details={"retry_after_s": retry_after})
            window.append(now)

    def tracked_keys(self) -> int:
        return len(self._windows)
