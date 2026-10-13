"""Tests for RRF fusion, including a hand-calculated example and property
tests (spec #19, #54)."""
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult
from rag_engine.retrieval.rrf import reciprocal_rank_fusion


def rr(chunk_id, doc_id, rank, score, method=RetrievalMethod.dense):
    return RetrievalResult(
        chunk_id=chunk_id, document_id=doc_id, content=f"content-{chunk_id}",
        score=score, rank=rank, method=method,
    )


def test_hand_calculated_rrf_fusion():
    """dense: [A(rank1), B(rank2)]; bm25: [B(rank1), C(rank2)]; k=60
    RRF(A) = 1/61
    RRF(B) = 1/62 + 1/61
    RRF(C) = 1/62
    => order should be B > A > C
    """
    dense = [rr("A", "d1", 1, 0.9), rr("B", "d1", 2, 0.8)]
    bm25 = [rr("B", "d1", 1, 15.0, RetrievalMethod.bm25), rr("C", "d2", 2, 10.0, RetrievalMethod.bm25)]

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)
    scores = {f.chunk_id: f.rrf_score for f in fused}

    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["C"] == pytest.approx(1 / 62)

    order = [f.chunk_id for f in fused]
    assert order == ["B", "A", "C"]


def test_rrf_explanation_contains_per_ranker_detail():
    dense = [rr("A", "d1", 1, 0.9)]
    bm25 = [rr("A", "d1", 3, 5.0, RetrievalMethod.bm25)]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)
    result = fused[0]
    reasons = result.reason()
    joined = " ".join(reasons)
    assert "dense" in joined and "bm25" in joined
    assert "rank 1" in joined
    assert "rank 3" in joined


def test_rrf_missing_from_one_ranker_not_penalized_to_zero():
    """A document appearing in only one ranker still gets a positive score,
    not zero -- it is simply not penalized for the ranker that never saw it."""
    dense = [rr("A", "d1", 1, 0.9)]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": []}, k=60)
    assert fused[0].rrf_score > 0


def test_weighted_rrf_changes_ranking():
    dense = [rr("A", "d1", 1, 0.9), rr("B", "d1", 2, 0.8)]
    bm25 = [rr("B", "d1", 1, 15.0, RetrievalMethod.bm25), rr("A", "d1", 2, 5.0, RetrievalMethod.bm25)]

    equal = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)
    assert equal[0].chunk_id in {"A", "B"}  # tie-ish, order depends on exact fractions

    bm25_heavy = reciprocal_rank_fusion(
        {"dense": dense, "bm25": bm25}, k=60, weights={"dense": 0.1, "bm25": 5.0}
    )
    assert bm25_heavy[0].chunk_id == "B"  # bm25's rank-1 pick should now dominate


def test_rrf_duplicate_within_single_ranker_not_double_counted():
    dense = [rr("A", "d1", 1, 0.9), rr("A", "d1", 1, 0.9)]  # duplicate entry
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": []}, k=60)
    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(1 / 61)  # counted once, not twice


def test_rrf_invalid_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": []}, k=0)


def test_rrf_top_k_truncates():
    dense = [rr(f"d{i}", "doc", i, 1.0 / i) for i in range(1, 20)]
    fused = reciprocal_rank_fusion({"dense": dense}, k=60, top_k=5)
    assert len(fused) == 5


def test_rrf_final_ranks_are_sequential():
    dense = [rr("A", "d1", 1, 0.9), rr("B", "d1", 2, 0.5)]
    fused = reciprocal_rank_fusion({"dense": dense}, k=60)
    assert [f.final_rank for f in fused] == [1, 2]


@given(
    ranks=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20, unique=True),
    k=st.integers(min_value=1, max_value=200),
)
def test_property_rrf_scores_always_positive_and_bounded(ranks, k):
    dense = [rr(f"c{i}", "d", rank, 1.0) for i, rank in enumerate(ranks)]
    fused = reciprocal_rank_fusion({"dense": dense}, k=k)
    for f in fused:
        assert f.rrf_score > 0
        assert f.rrf_score <= 1.0 / (k + 1)  # max possible contribution is rank=1


@given(
    n_rankers=st.integers(min_value=1, max_value=4),
)
def test_property_adding_a_ranking_cannot_decrease_an_existing_docs_score(n_rankers):
    """Adding another ranker's vote for a document already present can only
    add a positive contribution, never reduce the existing score."""
    dense = [rr("A", "d1", 1, 0.9)]
    baseline = reciprocal_rank_fusion({"dense": dense}, k=60)[0].rrf_score

    extra_rankers = {"dense": dense}
    for i in range(n_rankers):
        extra_rankers[f"extra{i}"] = [rr("A", "d1", 1, 1.0, RetrievalMethod.bm25)]
    boosted = reciprocal_rank_fusion(extra_rankers, k=60)[0].rrf_score

    assert boosted >= baseline
