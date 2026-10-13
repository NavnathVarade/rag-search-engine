"""From-scratch BM25 (Okapi BM25) scoring.

Standard formulation (Robertson & Sparck Jones / Robertson & Zaragoza):

    score(q, d) = sum_{t in q} IDF(t) * ( f(t,d) * (k1 + 1) )
                                        / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

  where:
    f(t,d)  = raw term frequency of t in document d
    |d|     = length of document d (token count)
    avgdl   = average document length across the corpus
    IDF(t)  = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
                (the "+1" inside the log is the modern BM25 variant that
                 keeps IDF non-negative even when a term appears in more
                 than half the corpus, unlike the classic Robertson/Sparck
                 Jones IDF which can go negative for very common terms)
    k1      = term-frequency saturation parameter (default 1.5)
    b       = length-normalization parameter, 0..1 (default 0.75)

Why BM25 can beat dense/semantic search on queries like
"PostgreSQL 16 CREATE INDEX CONCURRENTLY":
  Dense embeddings compress a chunk into a fixed-size vector optimized for
  *topical* similarity. Two chunks about "creating indexes in Postgres" can
  embed very close together whether they're about version 12 or version 16,
  and whether the command is CREATE INDEX or CREATE INDEX CONCURRENTLY --
  the embedding model was never trained to treat those token-level
  differences as semantically important, because most of the time they
  aren't. BM25, in contrast, rewards *exact token overlap* weighted by how
  rare/informative that token is corpus-wide (IDF). "CONCURRENTLY" and "16"
  are rare, high-IDF tokens across a general docs corpus, so a document
  containing them verbatim scores far higher than one merely "about
  indexing". This is precisely why the hybrid pipeline fuses both signals
  instead of picking one (see retrieval/rrf.py, docs/hybrid-search.md).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from rag_engine.retrieval.sparse.inverted_index import InvertedIndex


@dataclass(frozen=True)
class Bm25Hit:
    doc_id: str
    score: float
    matched_terms: dict[str, float] = field(default_factory=dict)  # term -> contribution


class Bm25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0, 1]")
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        n = self.index.document_count
        df = self.index.document_frequency(term)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score_document(self, query_terms: list[str], doc_id: str) -> tuple[float, dict[str, float]]:
        avgdl = self.index.average_document_length or 1.0
        doc_len = self.index.doc_length(doc_id) or 1
        total = 0.0
        contributions: dict[str, float] = {}
        for term in set(query_terms):
            f = self.index.term_frequency(term, doc_id)
            if f == 0:
                continue
            idf = self.idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            contribution = idf * (numerator / denominator)
            contributions[term] = contribution
            total += contribution
        return total, contributions

    def score(self, query: str, top_k: int = 50) -> list[Bm25Hit]:
        """Term-at-a-time scoring: IDF and avgdl are computed once per query and
        each posting list is walked once, so cost is O(total postings of the query
        terms) rather than O(candidates x corpus). Ties break on doc_id so results
        are deterministic."""
        query_terms = self.tokenizer_terms(query)
        if not query_terms or self.index.document_count == 0:
            return []

        avgdl = self.index.average_document_length or 1.0
        k1, b = self.k1, self.b
        idfs = {term: self.idf(term) for term in set(query_terms)}
        scores: dict[str, float] = {}
        for term, idf in idfs.items():
            for doc_id, f in self.index.postings_for(term).items():
                doc_len = self.index.doc_length(doc_id) or 1
                contribution = idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution

        top = heapq.nsmallest(top_k, ((-sc, doc_id) for doc_id, sc in scores.items() if sc > 0))
        hits: list[Bm25Hit] = []
        for neg_score, doc_id in top:
            _, contributions = self.score_document(query_terms, doc_id)
            hits.append(Bm25Hit(doc_id=doc_id, score=-neg_score, matched_terms=contributions))
        return hits

    def tokenizer_terms(self, query: str) -> list[str]:
        return self.index.tokenizer.tokenize(query)
