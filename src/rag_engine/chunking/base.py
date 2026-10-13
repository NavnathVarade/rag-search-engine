"""Chunk data model + Chunker protocol shared by every chunking strategy
(spec #9). A simple whitespace-based token estimator is used for token
budgets so chunking has no dependency on a specific tokenizer/model --
callers that need exact LLM token counts can substitute a real tokenizer
via `token_counter`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~= whitespace word count,
    the standard rule-of-thumb approximation before a real tokenizer is
    available). Documented as an estimate, not exact, in docs/chunking.md.
    """
    return len(text.split())


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    position: int
    token_count: int
    character_count: int
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)
    parent_chunk_id: str | None = None
    language: str | None = None
    metadata: dict = field(default_factory=dict)


class Chunker(Protocol):
    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]: ...
