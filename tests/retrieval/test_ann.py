"""Tests for brute-force exact search and HNSW approximate search,
including a measured recall@10 comparison of HNSW against brute-force
ground truth (spec #14, #55 -- numbers here are computed, not invented)."""
import numpy as np
import pytest

from rag_engine.retrieval.ann.brute_force import BruteForceIndex
from rag_engine.retrieval.ann.hnsw import HNSWIndex


def random_corpus(n: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim))
    ids = [f"doc_{i}" for i in range(n)]
    return ids, vectors, rng


def test_brute_force_finds_exact_nearest_neighbor():
    idx = BruteForceIndex(dimension=3)
    idx.add(["a", "b", "c"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]))
    results = idx.search(np.array([1.0, 0.0, 0.0]), top_k=2)
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][0] == "c"


def test_brute_force_add_remove():
    idx = BruteForceIndex(dimension=2)
    idx.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert idx.size == 2
    idx.remove(["a"])
    assert idx.size == 1
    results = idx.search(np.array([1.0, 0.0]), top_k=5)
    assert results[0][0] == "b"


def test_brute_force_top_k_capped_to_corpus_size():
    idx = BruteForceIndex(dimension=2)
    idx.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    results = idx.search(np.array([1.0, 1.0]), top_k=100)
    assert len(results) == 2


def test_brute_force_dimension_mismatch_raises():
    idx = BruteForceIndex(dimension=3)
    with pytest.raises(ValueError):
        idx.add(["a"], np.array([[1.0, 0.0]]))


def test_hnsw_exact_match_found_for_small_corpus():
    ids, vectors, _ = random_corpus(50, 16, seed=1)
    idx = HNSWIndex(dimension=16, m=8, ef_construction=100, ef_search=50)
    idx.add(ids, vectors)
    # query with one of the actual corpus vectors -> itself must be top-1
    query = vectors[10]
    results = idx.search(query, top_k=1)
    assert results[0][0] == "doc_10"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_hnsw_remove_excludes_from_results():
    ids, vectors, _ = random_corpus(30, 8, seed=2)
    idx = HNSWIndex(dimension=8, m=8, ef_construction=50, ef_search=30)
    idx.add(ids, vectors)
    idx.remove(["doc_5"])
    results = idx.search(vectors[5], top_k=30)
    returned_ids = {r[0] for r in results}
    assert "doc_5" not in returned_ids


@pytest.mark.slow
def test_hnsw_recall_at_10_against_brute_force_ground_truth():
    """Measures actual recall of HNSW vs brute-force on a random corpus.
    This is a real measurement executed by pytest, not a hardcoded claim."""
    n, dim, k = 500, 32, 10
    ids, vectors, rng = random_corpus(n, dim, seed=7)

    bf = BruteForceIndex(dimension=dim)
    bf.add(ids, vectors)

    hnsw = HNSWIndex(dimension=dim, m=16, ef_construction=200, ef_search=64, seed=7)
    hnsw.add(ids, vectors)

    n_queries = 20
    query_vectors = rng.normal(size=(n_queries, dim))

    total_recall = 0.0
    for q in query_vectors:
        gt = {doc_id for doc_id, _ in bf.search(q, top_k=k)}
        approx = {doc_id for doc_id, _ in hnsw.search(q, top_k=k)}
        total_recall += len(gt & approx) / k

    mean_recall = total_recall / n_queries
    print(f"\nMeasured HNSW recall@{k} vs brute force (n={n}, dim={dim}): {mean_recall:.3f}")
    # HNSW is approximate; assert a sane (not perfect) recall floor rather
    # than a fabricated exact number.
    assert mean_recall > 0.7


def _clustered(n: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(20, dim))
    x = centers[rng.integers(0, 20, n)] + 0.5 * rng.normal(size=(n, dim))
    q = centers[rng.integers(0, 20, 40)] + 0.5 * rng.normal(size=(40, dim))
    return [str(i) for i in range(n)], x, q


def test_hnsw_recall_on_clustered_data_is_high():
    ids, x, q = _clustered(1500, 48)
    brute = BruteForceIndex(48)
    brute.add(ids, x)
    hnsw = HNSWIndex(48, m=16, ef_construction=100, ef_search=64)
    hnsw.add(ids, x)
    recalls = [len({i for i, _ in brute.search(v, 10)} & {i for i, _ in hnsw.search(v, 10)}) / 10 for v in q]
    assert sum(recalls) / len(recalls) >= 0.95


def test_hnsw_remove_with_directed_edges_leaves_no_dangling_references():
    ids, x, _ = _clustered(300, 16)
    hnsw = HNSWIndex(16, m=8, ef_construction=50, ef_search=32)
    hnsw.add(ids, x)
    gone = set(ids[::3])
    hnsw.remove(list(gone))
    assert hnsw.size == len(ids) - len(gone)
    for node in hnsw._nodes.values():
        for layer in node.neighbors:
            assert not (layer & gone)
    hits = {i for i, _ in hnsw.search(x[1], 10)}
    assert hits and not (hits & gone)
    # removing the entry point must keep the index searchable
    hnsw.remove([hnsw._entry_point])
    assert hnsw.search(x[1], 5)
