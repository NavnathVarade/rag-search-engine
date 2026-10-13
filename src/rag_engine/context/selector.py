"""Context selection (spec #24-25): reranked candidates -> deduplication ->
diversity-aware, source-aware selection under a token budget -> final
ordered context with a human-readable explanation of why each chunk was
included (spec #21, #105).

Selection loop (greedy, in reranker-score order):
  1. skip a candidate that is a near-duplicate of something already
     selected (embedding cosine similarity, or lexical Jaccard fallback)
  2. skip a candidate once its document has already contributed
     `max_chunks_per_document` chunks (source diversity)
  3. skip a candidate that would exceed the token budget
  4. otherwise select it and record why
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rag_engine.context.budget import TokenBudget
from rag_engine.context.deduplicator import is_near_duplicate
from rag_engine.retrieval.models import RetrievalResult


@dataclass
class SelectedChunk:
    result: RetrievalResult
    reasons: list[str] = field(default_factory=list)


@dataclass
class ContextSelectionResult:
    selected: list[SelectedChunk]
    skipped: list[tuple[RetrievalResult, str]]  # (result, reason_skipped)
    total_tokens: int


class ContextSelector:
    def __init__(
        self,
        max_context_tokens: int = 6000,
        max_chunks_per_document: int = 3,
        dedup_threshold: float = 0.92,
        min_dense_similarity: float = 0.0,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_chunks_per_document = max_chunks_per_document
        self.dedup_threshold = dedup_threshold
        self.min_dense_similarity = min_dense_similarity

    def _has_evidence(self, c: RetrievalResult) -> bool:
        """Lexical match (BM25/TF-IDF) or a dense similarity above the threshold. Dense search always
        returns its top-k, however irrelevant; without this gate noise-level matches would count as evidence."""
        if self.min_dense_similarity <= 0:
            return True  # gate disabled
        if c.bm25_score is not None or c.method.value in ("bm25", "tfidf"):
            return True
        dense = c.dense_score if c.dense_score is not None else (c.score if c.method.value == "dense" else None)
        return dense is not None and dense >= self.min_dense_similarity

    def select(
        self,
        candidates: list[RetrievalResult],
        embeddings_by_chunk_id: dict[str, np.ndarray] | None = None,
    ) -> ContextSelectionResult:
        embeddings_by_chunk_id = embeddings_by_chunk_id or {}
        budget = TokenBudget(self.max_context_tokens)
        selected: list[SelectedChunk] = []
        skipped: list[tuple[RetrievalResult, str]] = []
        doc_counts: dict[str, int] = {}

        # candidates are assumed pre-sorted by relevance (reranker score,
        # or fused score if no reranker ran)
        for candidate in candidates:
            if not self._has_evidence(candidate):
                skipped.append((candidate, f"below evidence threshold (no lexical match, dense similarity < {self.min_dense_similarity})"))
                continue
            duplicate_of = self._find_duplicate(candidate, selected, embeddings_by_chunk_id)
            if duplicate_of is not None:
                skipped.append((candidate, f"near-duplicate of {duplicate_of}"))
                continue

            doc_id = candidate.document_id
            if doc_counts.get(doc_id, 0) >= self.max_chunks_per_document:
                skipped.append((candidate, f"source diversity cap reached for document {doc_id}"))
                continue

            if not budget.can_fit(candidate.content):
                skipped.append((candidate, "token budget exhausted"))
                continue

            reasons = [
                f"relevance score {candidate.score:.3f} (method={candidate.method.value})",
                f"document contribution {doc_counts.get(doc_id, 0) + 1}/{self.max_chunks_per_document}",
                f"fits within remaining budget ({budget.remaining} tokens left before this chunk)",
            ]
            budget.consume(candidate.content)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            selected.append(SelectedChunk(result=candidate, reasons=reasons))

        return ContextSelectionResult(selected=selected, skipped=skipped, total_tokens=budget.used_tokens)

    def _find_duplicate(
        self,
        candidate: RetrievalResult,
        selected: list[SelectedChunk],
        embeddings_by_chunk_id: dict[str, np.ndarray],
    ) -> str | None:
        cand_emb = embeddings_by_chunk_id.get(candidate.chunk_id)
        for s in selected:
            sel_emb = embeddings_by_chunk_id.get(s.result.chunk_id)
            if is_near_duplicate(
                candidate.content, s.result.content, cand_emb, sel_emb, threshold=self.dedup_threshold
            ):
                return s.result.chunk_id
        return None
