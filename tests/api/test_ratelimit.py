"""SlidingWindowLimiter unit tests with a controllable clock."""
import pytest

from rag_engine.api.ratelimit import SlidingWindowLimiter
from rag_engine.exceptions import RateLimitExceededError


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_allows_up_to_limit_then_blocks_then_recovers():
    clock = Clock()
    lim = SlidingWindowLimiter(window_s=60, clock=clock)
    for _ in range(3):
        lim.check("k", 3)
    with pytest.raises(RateLimitExceededError) as exc:
        lim.check("k", 3)
    assert exc.value.details["retry_after_s"] >= 1
    clock.t += 61
    lim.check("k", 3)  # window slid


def test_keys_are_independent():
    lim = SlidingWindowLimiter(clock=Clock())
    lim.check("a", 1)
    lim.check("b", 1)
    with pytest.raises(RateLimitExceededError):
        lim.check("a", 1)


def test_memory_is_bounded_by_lru_eviction():
    lim = SlidingWindowLimiter(max_keys=100, clock=Clock())
    for i in range(1000):
        lim.check(f"attacker-{i}", 5)
    assert lim.tracked_keys() <= 100
