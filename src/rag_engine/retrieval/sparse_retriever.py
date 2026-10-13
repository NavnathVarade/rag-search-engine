"""Wraps Bm25Scorer/TfidfScorer to return the common RetrievalResult
contract, mirroring DenseRetriever (spec #124)."""
from __future__ import annotations

from rag_engine.retrieval.dense import ChunkRecord
from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult
from rag_engine.retrieval.sparse.bm25 import Bm25Scorer
from rag_engine.retrieval.sparse.tfidf import TfidfScorer


class Bm25Retriever:
    def __init__(self, scorer: Bm25Scorer, chunk_store: dict[str, ChunkRecord]) -> None:
        self.scorer = scorer
        self.chunk_store = chunk_store

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        hits = self.scorer.score(query, top_k=top_k)
        results = []
        for rank, hit in enumerate(hits, start=1):
            record = self.chunk_store.get(hit.doc_id)
            if record is None:
                continue
            top_terms = sorted(hit.matched_terms.items(), key=lambda x: -x[1])[:3]
            terms_str = ", ".join(t for t, _ in top_terms)
            results.append(
                RetrievalResult(
                    chunk_id=hit.doc_id,
                    document_id=record.document_id,
                    content=record.content,
                    score=hit.score,
                    rank=rank,
                    method=RetrievalMethod.bm25,
                    metadata=record.metadata,
                    bm25_rank=rank,
                    bm25_score=hit.score,
                    explanation=[f"BM25 score {hit.score:.3f}; exact keyword match(es): {terms_str}"],
                )
            )
        return results


class TfidfRetriever:
    def __init__(self, scorer: TfidfScorer, chunk_store: dict[str, ChunkRecord]) -> None:
        self.scorer = scorer
        self.chunk_store = chunk_store

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        hits = self.scorer.score(query, top_k=top_k)
        results = []
        for rank, hit in enumerate(hits, start=1):
            record = self.chunk_store.get(hit.doc_id)
            if record is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=hit.doc_id,
                    document_id=record.document_id,
                    content=record.content,
                    score=hit.score,
                    rank=rank,
                    method=RetrievalMethod.tfidf,
                    metadata=record.metadata,
                    explanation=[f"TF-IDF score {hit.score:.3f}"],
                )
            )
        return results
