"""Normalized Discounted Cumulative Gain -- from scratch (spec #33).

    DCG@K  = sum_{i=1}^{K} rel_i / log2(i + 1)
    IDCG@K = DCG@K of the ideal ordering (relevant docs sorted by
             relevance first)
    NDCG@K = DCG@K / IDCG@K   (0 if IDCG@K is 0, i.e. no relevant docs)

Supports both binary relevance (relevant_ids: set[str]) and graded
relevance (relevance_scores: dict[str, float]) via `graded_ndcg_at_k`.
"""
from __future__ import annotations

import math


def dcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_ranking = list(relevant_ids)[:k]
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, len(ideal_ranking) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def graded_ndcg_at_k(retrieved_ids: list[str], relevance_scores: dict[str, float], k: int) -> float:
    dcg = sum(
        relevance_scores.get(doc_id, 0.0) / math.log2(i + 1)
        for i, doc_id in enumerate(retrieved_ids[:k], start=1)
    )
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(score / math.log2(i + 1) for i, score in enumerate(ideal_scores, start=1))
    return dcg / idcg if idcg > 0 else 0.0
