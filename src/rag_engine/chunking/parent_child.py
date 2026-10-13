"""Parent-child chunking (spec #9.G): large parent sections for LLM
context, small child chunks as the actual retrieval unit. Children carry
`parent_chunk_id`; context selection can expand a retrieved child out to
its parent's full text when more surrounding context is useful
(see context/selector.py).
"""
from __future__ import annotations

from rag_engine.chunking.base import Chunk
from rag_engine.chunking.recursive import RecursiveChunker


class ParentChildChunker:
    def __init__(self, parent_max_tokens: int = 1200, child_max_tokens: int = 250) -> None:
        self.parent_chunker = RecursiveChunker(max_tokens=parent_max_tokens, overlap_tokens=0)
        self.child_chunker = RecursiveChunker(max_tokens=child_max_tokens, overlap_tokens=20)

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        parents = self.parent_chunker.chunk(document_id, text, metadata)
        all_chunks: list[Chunk] = []
        for parent in parents:
            parent.chunk_id = f"{document_id}:parent:{parent.position}"
            all_chunks.append(parent)
            children = self.child_chunker.chunk(document_id, parent.content, metadata)
            for i, child in enumerate(children):
                child.chunk_id = f"{parent.chunk_id}:child:{i}"
                child.parent_chunk_id = parent.chunk_id
                all_chunks.append(child)
        return all_chunks
