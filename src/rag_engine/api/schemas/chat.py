"""Pydantic schemas for the chat/RAG API (spec #41)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from rag_engine.api.schemas.search import SearchResultItem, SearchTimings
from rag_engine.config import RetrievalMode


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    top_k: int = Field(default=50, ge=1, le=200)
    stream: bool = False


class CitationItem(BaseModel):
    marker: int
    chunk_id: str
    document_id: str | None
    source_label: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    retrieved_chunks: list[SearchResultItem]
    had_sufficient_evidence: bool
    injection_flags: list[str] = Field(default_factory=list)
    model: str
    prompt_tokens: int
    completion_tokens: int
    timings: SearchTimings
