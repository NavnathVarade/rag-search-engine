"""Tests for retrieval + generation evaluation metrics, with hand
calculated examples (spec #33-34, #54)."""
import math

import pytest

from rag_engine.evaluation.metrics.faithfulness import lexical_faithfulness_proxy
from rag_engine.evaluation.metrics.hallucination import unsupported_claim_rate_proxy
from rag_engine.evaluation.metrics.mrr import mean_reciprocal_rank, reciprocal_rank
from rag_engine.evaluation.metrics.ndcg import dcg_at_k, ndcg_at_k
from rag_engine.evaluation.metrics.recall import hit_rate_at_k, precision_at_k, recall_at_k


# --- Recall / Precision / Hit rate --------------------------------------
def test_recall_at_k_hand_calculated():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "e", "x"}  # x never retrieved
    # top 3 = {a,b,c}; hit = {c} => 1/3 of relevant set found
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)
    # top 5 = all -> hits {c, e} => 2/3
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_precision_at_k_hand_calculated():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)


def test_hit_rate_at_k():
    assert hit_rate_at_k(["a", "b"], {"b"}, k=2) == 1.0
    assert hit_rate_at_k(["a", "b"], {"z"}, k=2) == 0.0
    assert hit_rate_at_k(["a", "b"], set(), k=2) == 0.0


def test_recall_no_relevant_docs_returns_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


# --- MRR -----------------------------------------------------------
def test_reciprocal_rank_hand_calculated():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_mean_reciprocal_rank_hand_calculated():
    retrieved = [["a", "b"], ["x", "y", "z"]]
    relevant = [{"b"}, {"z"}]
    # RR1 = 1/2, RR2 = 1/3 => mean = (1/2 + 1/3)/2 = 5/12
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(5 / 12)


def test_mrr_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


# --- NDCG ----------------------------------------------------------
def test_dcg_hand_calculated():
    # retrieved: [a(rel), b(irrel), c(rel)]; relevant={a,c}
    # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
    dcg = dcg_at_k(["a", "b", "c"], {"a", "c"}, k=3)
    expected = 1 / math.log2(2) + 1 / math.log2(4)
    assert dcg == pytest.approx(expected)


def test_ndcg_perfect_ranking_equals_one():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_worst_ranking_less_than_perfect():
    perfect = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
    worst = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
    assert worst < perfect


def test_ndcg_no_relevant_docs_returns_zero():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


# --- Generation proxies ----------------------------------------------
def test_lexical_faithfulness_proxy_high_when_answer_grounded():
    context = ["Kubernetes restarts a failed container using restartPolicy settings."]
    answer = "Kubernetes restarts a failed container using restartPolicy settings."
    assert lexical_faithfulness_proxy(answer, context) == pytest.approx(1.0)


def test_lexical_faithfulness_proxy_low_when_answer_invents_content():
    context = ["Kubernetes restarts a failed container using restartPolicy settings."]
    answer = "Kubernetes deploys applications to the moon using quantum flux capacitors."
    score = lexical_faithfulness_proxy(answer, context)
    assert score < 0.3


def test_unsupported_claim_rate_is_inverse_of_faithfulness():
    context = ["some grounded fact about postgres indexes"]
    answer = "some grounded fact about postgres indexes"
    faithfulness = lexical_faithfulness_proxy(answer, context)
    hallucination = unsupported_claim_rate_proxy(answer, context)
    assert faithfulness + hallucination == pytest.approx(1.0)
