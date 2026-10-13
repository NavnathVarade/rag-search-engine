"""Exact brute-force nearest-neighbor search.

O(N*D) per query -- computes true cosine similarity against every stored
vector via a single vectorized matrix operation. This is the ground-truth
reference implementation used to measure HNSW's recall in benchmarks
(scripts/benchmark.py, docs/hnsw.md) since it always returns the exact
top-K by definition.
"""
from __future__ import annotations

import numpy as np

from rag_engine.retrieval.cosine import batch_cosine_similarity


class BruteForceIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._vectors: np.ndarray = np.zeros((0, dimension), dtype=np.float64)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected (N, {self.dimension}) vectors, got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")

        for doc_id in ids:
            if doc_id in self._id_to_row:
                self.remove([doc_id])

        start = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, vectors]) if self._vectors.size else vectors.copy()
        for offset, doc_id in enumerate(ids):
            self._id_to_row[doc_id] = start + offset
            self._ids.append(doc_id)

    def remove(self, ids: list[str]) -> None:
        remove_set = set(ids)
        if not remove_set:
            return
        keep_mask = [doc_id not in remove_set for doc_id in self._ids]
        self._vectors = self._vectors[np.array(keep_mask, dtype=bool)] if self._ids else self._vectors
        self._ids = [doc_id for doc_id, keep in zip(self._ids, keep_mask) if keep]
        self._id_to_row = {doc_id: i for i, doc_id in enumerate(self._ids)}

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.size == 0:
            return []
        sims = batch_cosine_similarity(query, self._vectors)
        top_k = min(top_k, sims.shape[0])
        # argpartition for O(N) selection, then sort just the top_k
        candidate_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        candidate_idx = candidate_idx[np.argsort(-sims[candidate_idx])]
        return [(self._ids[i], float(sims[i])) for i in candidate_idx]

    @property
    def size(self) -> int:
        return len(self._ids)
