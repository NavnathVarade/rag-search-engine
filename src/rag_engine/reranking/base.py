"""Reranker abstraction. Spec #22-23 requires the retriever to be broad
(top-50 recall-oriented) and the reranker to be selective (top-5,
precision-oriented) -- a bi-encoder (dense retrieval) embeds query and
document independently so document embeddings can be precomputed and
searched at scale, trading some accuracy for speed; a cross-encoder feeds
the (query, document) pair through the model *together*, letting every
query token attend to every document token, which is far more accurate
but cannot be precomputed -- it must run once per (query, candidate) pair
at query time. This is why reranking is applied only to a small candidate
set (typically 20-50) rather than the whole corpus (see docs/reranking.md).
"""
from __future__ import annotations

from typing import Protocol

from rag_engine.retrieval.models import RetrievalResult


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...
