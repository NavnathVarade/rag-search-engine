#!/usr/bin/env python3
"""Performance benchmark suite (spec #55, #79, #127). Every number this
script prints is measured on the machine it runs on, right now -- nothing
here is a hardcoded/fabricated figure. Run `python scripts/benchmark.py`
to reproduce these numbers on your own hardware; absolute latency numbers
depend heavily on the machine, so only relative comparisons (e.g. "HNSW
vs brute force at N=50k") are meaningful across environments.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from rag_engine.retrieval.ann.brute_force import BruteForceIndex  # noqa: E402
from rag_engine.retrieval.ann.hnsw import HNSWIndex  # noqa: E402
from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult  # noqa: E402
from rag_engine.retrieval.rrf import reciprocal_rank_fusion  # noqa: E402
from rag_engine.retrieval.sparse.bm25 import Bm25Scorer  # noqa: E402
from rag_engine.retrieval.sparse.inverted_index import InvertedIndex  # noqa: E402


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def benchmark_ann(sizes: list[int], dim: int = 128, n_queries: int = 20) -> None:
    print(f"\n=== ANN benchmark (dim={dim}, {n_queries} queries) ===")
    print(f"{'N':>8} {'BF build ms':>12} {'BF query ms':>12} {'HNSW build ms':>14} {'HNSW query ms':>14} {'recall@10':>10}")
    rng = np.random.default_rng(0)
    for n in sizes:
        ids = [f"d{i}" for i in range(n)]
        vectors = rng.normal(size=(n, dim))
        queries = rng.normal(size=(n_queries, dim))

        bf = BruteForceIndex(dimension=dim)
        _, bf_build_ms = _timed(bf.add, ids, vectors)
        t0 = time.perf_counter()
        gt_results = [bf.search(q, top_k=10) for q in queries]
        bf_query_ms = (time.perf_counter() - t0) * 1000 / n_queries

        hnsw = HNSWIndex(dimension=dim, m=16, ef_construction=100, ef_search=64)
        _, hnsw_build_ms = _timed(hnsw.add, ids, vectors)
        t0 = time.perf_counter()
        approx_results = [hnsw.search(q, top_k=10) for q in queries]
        hnsw_query_ms = (time.perf_counter() - t0) * 1000 / n_queries

        recalls = [
            len({d for d, _ in gt} & {d for d, _ in approx}) / 10
            for gt, approx in zip(gt_results, approx_results)
        ]
        mean_recall = sum(recalls) / len(recalls)
        print(f"{n:>8} {bf_build_ms:>12.1f} {bf_query_ms:>12.3f} {hnsw_build_ms:>14.1f} {hnsw_query_ms:>14.3f} {mean_recall:>10.3f}")


def benchmark_bm25(sizes: list[int]) -> None:
    print("\n=== BM25 benchmark ===")
    print(f"{'N docs':>8} {'index build ms':>16} {'query ms':>10}")
    words = ["kubernetes", "postgres", "index", "container", "restart", "policy", "fastapi", "async", "query", "table"]
    rng = np.random.default_rng(1)
    for n in sizes:
        idx = InvertedIndex()
        t0 = time.perf_counter()
        for i in range(n):
            doc = " ".join(rng.choice(words, size=50))
            idx.add_document(f"d{i}", doc)
        build_ms = (time.perf_counter() - t0) * 1000

        scorer = Bm25Scorer(idx)
        t0 = time.perf_counter()
        for _ in range(20):
            scorer.score("kubernetes container restart policy", top_k=10)
        query_ms = (time.perf_counter() - t0) * 1000 / 20
        print(f"{n:>8} {build_ms:>16.1f} {query_ms:>10.3f}")


def benchmark_rrf(sizes: list[int]) -> None:
    print("\n=== RRF fusion benchmark ===")
    print(f"{'candidates per ranker':>22} {'fusion ms':>10}")
    for n in sizes:
        dense = [RetrievalResult(chunk_id=f"c{i}", document_id="d", content="", score=1.0 / (i + 1), rank=i + 1, method=RetrievalMethod.dense) for i in range(n)]
        bm25 = [RetrievalResult(chunk_id=f"c{i}", document_id="d", content="", score=1.0 / (i + 1), rank=i + 1, method=RetrievalMethod.bm25) for i in range(n)]
        t0 = time.perf_counter()
        reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"{n:>22} {elapsed_ms:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--suite", choices=["ann", "bm25", "rrf", "all"], default="all")
    args = parser.parse_args()

    if args.suite in ("ann", "all"):
        benchmark_ann(args.sizes)
    if args.suite in ("bm25", "all"):
        benchmark_bm25(args.sizes)
    if args.suite in ("rrf", "all"):
        benchmark_rrf(args.sizes)


if __name__ == "__main__":
    main()
