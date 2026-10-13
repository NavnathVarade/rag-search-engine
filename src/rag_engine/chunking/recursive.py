"""Recursive chunking: split by a hierarchy of separators, per spec #9.B --
headings -> paragraphs -> sentences -> words -- recursing into a piece
only if it still exceeds the token budget after splitting at the current
level. This is the classic "recursive character/text splitter" strategy,
implemented here without any external text-splitting library.
"""
from __future__ import annotations

import re

from rag_engine.chunking.base import Chunk, estimate_tokens

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_SEPARATOR_HIERARCHY = [
    "\n\n\n",  # section-ish breaks
    "\n\n",  # paragraphs
    "\n",  # lines
    None,  # sentence split (handled specially)
    " ",  # words -- last resort
]


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class RecursiveChunker:
    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 50) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _split(self, text: str, level: int = 0) -> list[str]:
        if estimate_tokens(text) <= self.max_tokens or level >= len(_SEPARATOR_HIERARCHY):
            return [text] if text.strip() else []

        sep = _SEPARATOR_HIERARCHY[level]
        pieces = _split_sentences(text) if sep is None else [p for p in text.split(sep) if p.strip()]

        if len(pieces) <= 1:
            return self._split(text, level + 1)

        results: list[str] = []
        for piece in pieces:
            results.extend(self._split(piece, level + 1) if estimate_tokens(piece) > self.max_tokens else [piece])
        return results

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily pack small pieces together up to max_tokens, adding a
        trailing-word overlap between consecutive merged chunks."""
        merged: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece in pieces:
            piece_tokens = estimate_tokens(piece)
            if current and current_tokens + piece_tokens > self.max_tokens:
                merged.append(" ".join(current))
                if self.overlap_tokens > 0:
                    overlap_words = " ".join(current).split()[-self.overlap_tokens :]
                    current = [" ".join(overlap_words)] if overlap_words else []
                    current_tokens = len(overlap_words)
                else:
                    current, current_tokens = [], 0
            current.append(piece)
            current_tokens += piece_tokens
        if current:
            merged.append(" ".join(current))
        return merged

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text.strip():
            return []
        raw_pieces = self._split(text)
        merged = self._merge_with_overlap(raw_pieces)

        chunks: list[Chunk] = []
        for position, content in enumerate(merged):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:recursive:{position}",
                    document_id=document_id,
                    content=content,
                    position=position,
                    token_count=estimate_tokens(content),
                    character_count=len(content),
                    metadata=dict(metadata or {}),
                )
            )
        return chunks
