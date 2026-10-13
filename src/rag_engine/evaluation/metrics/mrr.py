"""Mean Reciprocal Rank -- from scratch (spec #33).

RR(q) = 1 / rank_of_first_relevant_result  (0 if no relevant result appears
at all in the ranked list). MRR is the mean of RR across queries.
"""
from __future__ import annotations


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(all_retrieved: list[list[str]], all_relevant: list[set[str]]) -> float:
    if not all_retrieved:
        return 0.0
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("all_retrieved and all_relevant must have the same length")
    scores = [reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(scores) / len(scores)
