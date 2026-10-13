"""Deterministic, offline lexical-overlap reranker used as the default in
development/CI/this sandbox (no network access to download a real
cross-encoder model here). This is explicitly NOT a neural cross-encoder
-- it approximates "does this candidate actually address the query" via a
transparent, hand-computed signal (query-term coverage + BM25-style term
weighting recomputed over just the candidate set) so the reranking *stage*
of the pipeline (context/selection after top-50 -> top-5) is real,
testable, and runs end-to-end without a model download. See
cross_encoder.py for the real neural implementation used in production.
"""
from __future__ import annotations

import math

from rag_engine.retrieval.models import RetrievalResult
from rag_engine.retrieval.sparse.tokenizer import Tokenizer


class HeuristicReranker:
    """Rescoring via query-term coverage weighted by inverse candidate-set
    document frequency (a mini BM25 recomputed only over the retrieved
    candidates, not the full corpus) plus an exact-phrase-match bonus."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or Tokenizer()

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return candidates[:top_k]

        candidate_tokens = [self.tokenizer.tokenize(c.content) for c in candidates]
        n = len(candidates)
        df = {}
        for term in set(query_terms):
            df[term] = sum(1 for toks in candidate_tokens if term in toks)

        scored: list[tuple[float, RetrievalResult]] = []
        query_lower = query.lower()
        for candidate, tokens in zip(candidates, candidate_tokens):
            token_set = set(tokens)
            score = 0.0
            for term in set(query_terms):
                if term in token_set:
                    idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
                    tf = tokens.count(term)
                    score += idf * math.log(1 + tf)
            if query_lower in candidate.content.lower():
                score += 2.0  # exact phrase match bonus
            coverage = len(token_set & set(query_terms)) / len(set(query_terms))
            score *= (0.5 + 0.5 * coverage)
            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = []
        for new_rank, (score, candidate) in enumerate(scored[:top_k], start=1):
            candidate.reranker_score = score
            candidate.explanation = [
                *candidate.explanation,
                f"Reranker score {score:.3f} (query-term coverage + phrase-match heuristic)",
            ]
            candidate.rank = new_rank
            reranked.append(candidate)
        return reranked
