"""Hybrid retrieval orchestrator: implements the five retrieval modes from
spec #20 (dense-only, BM25-only, TF-IDF baseline, hybrid RRF, hybrid
weighted RRF) behind one call so the API/CLI can select a mode per query.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_engine.config import RetrievalMode
from rag_engine.retrieval.dense import DenseRetriever
from rag_engine.retrieval.models import RetrievalResult
from rag_engine.retrieval.rrf import reciprocal_rank_fusion
from rag_engine.retrieval.sparse_retriever import Bm25Retriever, TfidfRetriever


@dataclass
class HybridSearchTimings:
    embedding_ms: float = 0.0
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class HybridSearchResponse:
    query: str
    retrieval_mode: str
    results: list[RetrievalResult]
    timings: HybridSearchTimings = field(default_factory=HybridSearchTimings)
    degraded: list[str] = field(default_factory=list)  # e.g. ["bm25_unavailable"]


class HybridRetriever:
    """Combines a DenseRetriever, Bm25Retriever, and (optionally) a
    TfidfRetriever. Any component may be None, in which case that mode
    degrades gracefully (spec #104) -- e.g. hybrid search with no dense
    retriever configured silently falls back to BM25-only and reports the
    degradation in HybridSearchResponse.degraded.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None,
        bm25_retriever: Bm25Retriever | None,
        tfidf_retriever: TfidfRetriever | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.tfidf_retriever = tfidf_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.hybrid,
        top_k: int = 50,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> HybridSearchResponse:
        timings = HybridSearchTimings()
        degraded: list[str] = []
        t_start = time.perf_counter()
        dense_top_k = dense_top_k or top_k
        sparse_top_k = sparse_top_k or top_k

        if mode == RetrievalMode.dense:
            results = self._run_dense(query, dense_top_k, timings)
        elif mode == RetrievalMode.bm25:
            results = self._run_bm25(query, sparse_top_k, timings)
        elif mode == RetrievalMode.tfidf:
            results = self._run_tfidf(query, sparse_top_k)
        elif mode in (RetrievalMode.hybrid, RetrievalMode.hybrid_weighted):
            dense_results = self._run_dense(query, dense_top_k, timings) if self.dense_retriever else []
            bm25_results = self._run_bm25(query, sparse_top_k, timings) if self.bm25_retriever else []
            if not self.dense_retriever:
                degraded.append("dense_unavailable_falling_back_to_bm25_only")
            if not self.bm25_retriever:
                degraded.append("bm25_unavailable_falling_back_to_dense_only")

            t_fusion = time.perf_counter()
            weights = (
                {"dense": self.dense_weight, "bm25": self.sparse_weight}
                if mode == RetrievalMode.hybrid_weighted
                else None
            )
            fused = reciprocal_rank_fusion(
                {"dense": dense_results, "bm25": bm25_results}, k=self.rrf_k, weights=weights, top_k=top_k
            )
            results = [f.to_retrieval_result() for f in fused]
            timings.fusion_ms = (time.perf_counter() - t_fusion) * 1000
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        timings.total_ms = (time.perf_counter() - t_start) * 1000
        return HybridSearchResponse(query=query, retrieval_mode=mode.value, results=results, timings=timings, degraded=degraded)

    def _run_dense(self, query: str, top_k: int, timings: HybridSearchTimings) -> list[RetrievalResult]:
        if not self.dense_retriever:
            return []
        t0 = time.perf_counter()
        results = self.dense_retriever.retrieve(query, top_k=top_k)
        timings.dense_ms = (time.perf_counter() - t0) * 1000
        return results

    def _run_bm25(self, query: str, top_k: int, timings: HybridSearchTimings) -> list[RetrievalResult]:
        if not self.bm25_retriever:
            return []
        t0 = time.perf_counter()
        results = self.bm25_retriever.retrieve(query, top_k=top_k)
        timings.bm25_ms = (time.perf_counter() - t0) * 1000
        return results

    def _run_tfidf(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.tfidf_retriever:
            return []
        return self.tfidf_retriever.retrieve(query, top_k=top_k)
