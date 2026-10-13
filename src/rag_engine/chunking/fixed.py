"""Fixed-size chunking with configurable overlap (spec #9.A).

Splits on whitespace-delimited "words" to avoid cutting inside a word,
and slides a window of `chunk_size` words forward by `chunk_size - overlap`
each step. Overlap exists so a sentence/fact split across a chunk boundary
still appears whole in at least one chunk; too much overlap wastes index
space and inflates near-duplicate content in retrieval (see docs/chunking.md
"Why overlap becomes harmful").
"""
from __future__ import annotations

from rag_engine.chunking.base import Chunk, estimate_tokens
from rag_engine.exceptions import ChunkingError


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ChunkingError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        position = 0
        i = 0
        while i < len(words):
            window = words[i : i + self.chunk_size]
            content = " ".join(window)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:fixed:{position}",
                    document_id=document_id,
                    content=content,
                    position=position,
                    token_count=estimate_tokens(content),
                    character_count=len(content),
                    metadata=dict(metadata or {}),
                )
            )
            position += 1
            if i + self.chunk_size >= len(words):
                break
            i += step
        return chunks
