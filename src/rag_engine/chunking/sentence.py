"""Sentence-boundary chunking respecting a token budget (spec #9.C)."""
from __future__ import annotations

from rag_engine.chunking.base import Chunk, estimate_tokens
from rag_engine.chunking.recursive import _split_sentences


class SentenceChunker:
    def __init__(self, max_tokens: int = 300, overlap_sentences: int = 1) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        position = 0
        i = 0
        while i < len(sentences):
            window: list[str] = []
            tokens = 0
            j = i
            while j < len(sentences) and (not window or tokens + estimate_tokens(sentences[j]) <= self.max_tokens):
                tokens += estimate_tokens(sentences[j])
                window.append(sentences[j])
                j += 1
            content = " ".join(window)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:sentence:{position}",
                    document_id=document_id,
                    content=content,
                    position=position,
                    token_count=estimate_tokens(content),
                    character_count=len(content),
                    metadata=dict(metadata or {}),
                )
            )
            position += 1
            i = max(j - self.overlap_sentences, i + 1)
        return chunks
