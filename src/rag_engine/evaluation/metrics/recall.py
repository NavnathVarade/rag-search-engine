"""Recall@K, Precision@K, Hit Rate -- from scratch (spec #33)."""
from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the relevant set found within the top-K retrieved ids.
    Undefined (returns 0.0) if there are no relevant docs at all."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-K retrieved ids that are actually relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if at least one relevant doc appears in the top-K, else 0.0."""
    if not relevant_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0
