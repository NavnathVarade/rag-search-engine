"""Shared retrieval result contract (spec #124-126).

Every retriever (dense, BM25, TF-IDF, hybrid/RRF) and the reranker return
RetrievalResult objects with a compatible shape so that the evaluation
framework, API layer, and explanation builder never need to special-case
which retriever produced a hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalMethod(str, Enum):
    dense = "dense"
    bm25 = "bm25"
    tfidf = "tfidf"
    hybrid_rrf = "hybrid_rrf"
    reranked = "reranked"


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    method: RetrievalMethod
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    # Optional per-signal diagnostics populated by hybrid/rerank stages
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "method": self.method.value,
            "metadata": self.metadata,
            "explanation": self.explanation,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "reranker_score": self.reranker_score,
        }
