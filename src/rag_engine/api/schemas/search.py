"""Pydantic request/response schemas for the search API (spec #39-40)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from rag_engine.config import RetrievalMode


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    top_k: int = Field(default=10, ge=1, le=200)
    debug: bool = False


class RankerDetail(BaseModel):
    rank: int | None = None
    score: float | None = None


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    rank: int
    score: float
    method: str
    dense: RankerDetail | None = None
    bm25: RankerDetail | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    explanation: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SearchTimings(BaseModel):
    embedding_ms: float = 0.0
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0


class SearchResponse(BaseModel):
    query: str
    retrieval_mode: str
    results: list[SearchResultItem]
    timings: SearchTimings
    degraded: list[str] = Field(default_factory=list)
