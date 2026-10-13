"""From-scratch HNSW (Hierarchical Navigable Small World) approximate
nearest-neighbor index, following Malkov & Yashunin, "Efficient and
Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable
Small World Graphs" (2016/2018), https://arxiv.org/abs/1603.09320.

Concept (see docs/hnsw.md for full explanation with diagrams):
  * A multi-layer graph is built where layer 0 contains every point and
    each higher layer contains an exponentially-shrinking random subset
    (probability ~ 1/ln(M)), acting like a skip-list over the graph.
  * Each node keeps up to M neighbor edges per layer (M0 = 2*M at layer 0,
    matching the reference paper's convention).
  * Insertion: starting from the top layer's entry point, greedily
    descend layer by layer to find the closest node, then at the node's
    assigned layer and below, run a best-first search of width
    `ef_construction` to select M neighbors to connect (heuristic:
    closest-M, matching the paper's simple neighbor-selection option).
  * Search: same greedy descent from the top, then a best-first search of
    width `ef_search` at layer 0, returning the top_k closest candidates
    found.

Trade-offs (measured, not assumed -- see scripts/benchmark.py and
docs/hnsw.md "Benchmarks" section for how to reproduce numbers on your
own machine/corpus, since we do not fabricate hardware-dependent
recall/latency figures here):
  * Approximate: unlike BruteForceIndex, HNSW is not guaranteed to return
    the true top-K -- recall depends on M / ef_construction / ef_search.
  * Higher ef_search -> higher recall, higher query latency.
  * Higher M / ef_construction -> better graph quality (higher recall at
    a given ef_search) but more memory and slower index builds.
  * Brute force is O(N) per query with no memory overhead beyond the raw
    vectors; HNSW trades index memory (graph edges) and build time for
    sub-linear query time at scale.
"""
from __future__ import annotations

import heapq
import math
import random

import numpy as np


class _Node:
    __slots__ = ("doc_id", "vector", "level", "neighbors")

    def __init__(self, doc_id: str, vector: np.ndarray, level: int) -> None:
        self.doc_id = doc_id
        self.vector = vector
        self.level = level
        # neighbors[layer] = set of doc_ids
        self.neighbors: list[set[str]] = [set() for _ in range(level + 1)]


class HNSWIndex:
    def __init__(
        self,
        dimension: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
    ) -> None:
        self.dimension = dimension
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._level_lambda = 1.0 / math.log(m) if m > 1 else 1.0
        self._rng = random.Random(seed)

        self._nodes: dict[str, _Node] = {}
        self._entry_point: str | None = None
        self._max_level = -1

    # -- helpers ----------------------------------------------------------
    def _random_level(self) -> int:
        # Exponentially decaying level assignment, per the paper.
        return int(-math.log(self._rng.random() + 1e-12) * self._level_lambda)

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        """Unit-normalize once at insert/query time so every graph hop is a
        single dot product (cosine of unit vectors). Zero vectors stay zero,
        giving similarity 0.0 -- the same convention as cosine_similarity()."""
        v = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-12 else v

    def _sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _search_layer(self, query: np.ndarray, entry_points: set[str], layer: int, ef: int) -> list[tuple[float, str]]:
        """Best-first search on a single layer. Returns list of (similarity, doc_id)
        sorted descending, of length <= ef."""
        visited: set[str] = set(entry_points)
        candidates: list[tuple[float, str]] = []  # max-heap via negation
        results: list[tuple[float, str]] = []  # min-heap (worst-of-best at top)

        for ep in entry_points:
            sim = self._sim(query, self._nodes[ep].vector)
            heapq.heappush(candidates, (-sim, ep))
            heapq.heappush(results, (sim, ep))

        while candidates:
            neg_sim, current_id = heapq.heappop(candidates)
            current_sim = -neg_sim
            worst_result_sim = results[0][0] if results else -math.inf
            if current_sim < worst_result_sim and len(results) >= ef:
                break

            for neighbor_id in self._nodes[current_id].neighbors[layer]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbor_sim = self._sim(query, self._nodes[neighbor_id].vector)
                worst_result_sim = results[0][0] if results else -math.inf
                if len(results) < ef or neighbor_sim > worst_result_sim:
                    heapq.heappush(candidates, (-neighbor_sim, neighbor_id))
                    heapq.heappush(results, (neighbor_sim, neighbor_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted(results, key=lambda x: -x[0])

    def _select_neighbors(self, candidates: list[tuple[float, str]], m: int) -> set[str]:
        """Heuristic neighbor selection (HNSW paper, Algorithm 4).

        Walk candidates from most to least similar to the base point and keep a
        candidate only if it is closer to the base than to every neighbor already
        kept. This favors diverse directions over a tight cluster of mutual
        near-duplicates, which keeps the graph navigable. Remaining slots are
        back-filled with the closest discarded candidates ("keepPrunedConnections")
        so nodes are never left under-connected.
        """
        ordered = sorted(candidates, key=lambda x: -x[0])
        selected: list[tuple[float, str]] = []
        discarded: list[tuple[float, str]] = []
        for sim_to_base, cand_id in ordered:
            if len(selected) >= m:
                break
            cand_vec = self._nodes[cand_id].vector
            if all(sim_to_base > self._sim(cand_vec, self._nodes[sel_id].vector) for _, sel_id in selected):
                selected.append((sim_to_base, cand_id))
            else:
                discarded.append((sim_to_base, cand_id))
        for item in discarded:
            if len(selected) >= m:
                break
            selected.append(item)
        return {cand_id for _, cand_id in selected}

    # -- mutation -----------------------------------------------------------
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected (N, {self.dimension}) vectors, got {vectors.shape}")
        for doc_id, vector in zip(ids, vectors):
            self._insert(doc_id, vector)

    def _insert(self, doc_id: str, vector: np.ndarray) -> None:
        if doc_id in self._nodes:
            self.remove([doc_id])

        vector = self._unit(vector)
        level = self._random_level()
        node = _Node(doc_id, vector, level)
        self._nodes[doc_id] = node

        if self._entry_point is None:
            self._entry_point = doc_id
            self._max_level = level
            return

        entry = self._entry_point
        # Descend from top to level+1 with ef=1 (greedy single path)
        for layer in range(self._max_level, level, -1):
            nearest = self._search_layer(vector, {entry}, layer, ef=1)
            if nearest:
                entry = nearest[0][1]

        entry_points = {entry}
        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(vector, entry_points, layer, ef=self.ef_construction)
            m_layer = self.m0 if layer == 0 else self.m
            selected = self._select_neighbors(candidates, m_layer)
            node.neighbors[layer] = set(selected)
            for neighbor_id in selected:
                neighbor = self._nodes[neighbor_id]
                neighbor.neighbors[layer].add(doc_id)
                if len(neighbor.neighbors[layer]) > m_layer:
                    # Edges are DIRECTED, as in the HNSW paper: shrinking this
                    # neighbor's list must not delete the reverse edges other
                    # nodes hold toward it, or those nodes lose connectivity.
                    # remove() scans for in-edges, so dangling edges cannot occur.
                    resc = [(self._sim(neighbor.vector, self._nodes[nid].vector), nid) for nid in neighbor.neighbors[layer]]
                    neighbor.neighbors[layer] = self._select_neighbors(resc, m_layer)
            entry_points = {doc_id for _, doc_id in candidates} or entry_points

        if level > self._max_level:
            self._max_level = level
            self._entry_point = doc_id

    def remove(self, ids: list[str]) -> None:
        to_remove = {i for i in ids if i in self._nodes}
        if not to_remove:
            return
        for doc_id in to_remove:
            del self._nodes[doc_id]
        # Graph edges are directed, so in-edges can come from any node: scan.
        for node in self._nodes.values():
            for neighbor_set in node.neighbors:
                neighbor_set -= to_remove
        if self._entry_point in to_remove:
            self._entry_point = max(self._nodes, key=lambda i: self._nodes[i].level, default=None)
            self._max_level = self._nodes[self._entry_point].level if self._entry_point else -1

    # -- search -----------------------------------------------------------
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._entry_point is None:
            return []
        query = self._unit(query)
        entry = self._entry_point
        for layer in range(self._max_level, 0, -1):
            nearest = self._search_layer(query, {entry}, layer, ef=1)
            if nearest:
                entry = nearest[0][1]

        ef = max(self.ef_search, top_k)
        results = self._search_layer(query, {entry}, 0, ef=ef)
        return [(doc_id, sim) for sim, doc_id in results[:top_k]]

    @property
    def size(self) -> int:
        return len(self._nodes)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_level": self._max_level,
            "m": self.m,
            "m0": self.m0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
