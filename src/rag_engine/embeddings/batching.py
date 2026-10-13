"""Batching helper for embedding calls -- splits large text lists into
fixed-size batches so providers (local model or remote API) never receive
an unbounded batch (spec #11 batching requirement)."""
from __future__ import annotations

from collections.abc import Iterator


def batched(items: list[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
