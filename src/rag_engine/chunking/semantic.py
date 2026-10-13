"""Semantic chunking (spec #9.D): splits text into sentences, embeds each
sentence, and inserts a chunk boundary wherever the cosine similarity
between consecutive sentences drops below a threshold (a topic-transition
signal), then packs sentences between boundaries into token-budgeted
chunks. Requires an EmbeddingProvider -- this is the most computationally
expensive strategy (one embedding call per sentence) which is precisely
why docs/chunking.md flags it as the strategy to reach for only when
topic drift within documents is a known problem, not a default.
"""
from __future__ import annotations

import numpy as np

from rag_engine.chunking.base import Chunk, estimate_tokens
from rag_engine.chunking.recursive import _split_sentences
from rag_engine.embeddings.base import EmbeddingProvider
from rag_engine.retrieval.cosine import cosine_similarity


class SemanticChunker:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        similarity_threshold: float = 0.5,
        max_tokens: int = 400,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens

    def chunk(self, document_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return self._finalize(document_id, [sentences], metadata)

        embeddings: np.ndarray = self.embedder.embed_documents(sentences)

        groups: list[list[str]] = [[sentences[0]]]
        group_tokens = estimate_tokens(sentences[0])
        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            sentence_tokens = estimate_tokens(sentences[i])
            topic_shift = sim < self.similarity_threshold
            over_budget = group_tokens + sentence_tokens > self.max_tokens
            if topic_shift or over_budget:
                groups.append([sentences[i]])
                group_tokens = sentence_tokens
            else:
                groups[-1].append(sentences[i])
                group_tokens += sentence_tokens

        return self._finalize(document_id, groups, metadata)

    def _finalize(self, document_id: str, groups: list[list[str]], metadata: dict | None) -> list[Chunk]:
        chunks = []
        for position, group in enumerate(groups):
            content = " ".join(group)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:semantic:{position}",
                    document_id=document_id,
                    content=content,
                    position=position,
                    token_count=estimate_tokens(content),
                    character_count=len(content),
                    metadata=dict(metadata or {}),
                )
            )
        return chunks
