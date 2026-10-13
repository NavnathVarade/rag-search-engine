"""Token budget accounting for context selection (spec #25)."""
from __future__ import annotations

from rag_engine.chunking.base import estimate_tokens


class TokenBudget:
    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def can_fit(self, text: str) -> bool:
        return self.used_tokens + estimate_tokens(text) <= self.max_tokens

    def consume(self, text: str) -> None:
        self.used_tokens += estimate_tokens(text)

    @property
    def remaining(self) -> int:
        return max(self.max_tokens - self.used_tokens, 0)
