"""LLM generation abstraction (spec #26). Core pipeline code depends only
on this Protocol, never on a specific vendor SDK (spec #115)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str = "stop"


class LLMProvider(Protocol):
    async def generate(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.1, max_tokens: int = 1000
    ) -> GenerationResult: ...

    # Declared as a plain `def` returning AsyncIterator: implementations are async
    # generators (`async def` + `yield`), which is what this signature describes.
    def generate_stream(
        self, system_prompt: str, user_prompt: str, *, temperature: float = 0.1, max_tokens: int = 1000
    ) -> AsyncIterator[str]: ...

    def model_name(self) -> str: ...
