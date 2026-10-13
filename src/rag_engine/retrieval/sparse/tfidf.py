"""From-scratch TF-IDF scoring, used as a baseline retrieval mode.

    tf(t, d)  = raw term count of t in document d
    idf(t)    = ln( (1 + N) / (1 + df(t)) ) + 1        (smoothed IDF, avoids
                                                          division by zero
                                                          and zero-IDF for
                                                          terms in every doc)
    score(q,d) = sum_{t in q} tf(t,d) * idf(t)

This is intentionally the classic smoothed-IDF formulation (as used by
scikit-learn's TfidfVectorizer) so results are interpretable/comparable,
while being implemented here without scikit-learn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rag_engine.retrieval.sparse.inverted_index import InvertedIndex


@dataclass(frozen=True)
class TfidfHit:
    doc_id: str
    score: float
    matched_terms: dict[str, float]  # term -> contribution


class TfidfScorer:
    def __init__(self, index: InvertedIndex) -> None:
        self.index = index

    def idf(self, term: str) -> float:
        n = self.index.document_count
        df = self.index.document_frequency(term)
        return math.log((1 + n) / (1 + df)) + 1.0

    def score(self, query: str, top_k: int = 50) -> list[TfidfHit]:
        query_terms = self.index.tokenizer.tokenize(query)
        if not query_terms or self.index.document_count == 0:
            return []

        candidate_docs: set[str] = set()
        for term in set(query_terms):
            candidate_docs |= self.index.matching_documents(term)

        results: list[TfidfHit] = []
        for doc_id in candidate_docs:
            contributions: dict[str, float] = {}
            total = 0.0
            for term in set(query_terms):
                tf = self.index.term_frequency(term, doc_id)
                if tf == 0:
                    continue
                contribution = tf * self.idf(term)
                contributions[term] = contribution
                total += contribution
            if total > 0:
                results.append(TfidfHit(doc_id=doc_id, score=total, matched_terms=contributions))

        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k]
