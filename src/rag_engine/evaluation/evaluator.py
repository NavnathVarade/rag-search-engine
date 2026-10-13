"""Retrieval evaluator: runs a HybridRetriever (or any callable retriever)
against an evaluation dataset and computes real, aggregated Recall@K /
Precision@K / MRR / NDCG / Hit Rate (spec #33). No numbers here are ever
invented -- every metric is computed from the retriever's actual output
against the dataset's `relevant_chunks` labels.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from rag_engine.evaluation.dataset import EvaluationQuestion
from rag_engine.evaluation.metrics.mrr import mean_reciprocal_rank
from rag_engine.evaluation.metrics.ndcg import ndcg_at_k
from rag_engine.evaluation.metrics.recall import hit_rate_at_k, precision_at_k, recall_at_k

_K_VALUES = (1, 3, 5, 10, 20, 50)


@dataclass
class PerQuestionResult:
    question_id: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: set[str]
    latency_ms: float


@dataclass
class RetrievalEvaluationReport:
    n_questions: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    mean_latency_ms: float
    per_question: list[PerQuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "mean_latency_ms": self.mean_latency_ms,
        }


def evaluate_retrieval(
    questions: list[EvaluationQuestion],
    retrieve_fn: Callable[[str, int], tuple[list[Any], float]],  # (query, k) -> (results, latency_ms)
    k_values: tuple[int, ...] = _K_VALUES,
    max_k: int = 50,
) -> RetrievalEvaluationReport:
    if not questions:
        raise ValueError("cannot evaluate an empty question set")

    per_question: list[PerQuestionResult] = []
    for q in questions:
        results, latency_ms = retrieve_fn(q.question, max_k)
        retrieved_ids = [r.chunk_id for r in results]
        per_question.append(
            PerQuestionResult(
                question_id=q.id,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=set(q.relevant_chunks),
                latency_ms=latency_ms,
            )
        )

    recall = {k: _mean(recall_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in per_question) for k in k_values}
    precision = {k: _mean(precision_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in per_question) for k in k_values}
    hit_rate = {k: _mean(hit_rate_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in per_question) for k in k_values}
    ndcg = {k: _mean(ndcg_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in per_question) for k in k_values}
    mrr = mean_reciprocal_rank([r.retrieved_chunk_ids for r in per_question], [r.relevant_chunk_ids for r in per_question])
    mean_latency = _mean(r.latency_ms for r in per_question)

    return RetrievalEvaluationReport(
        n_questions=len(questions), recall_at_k=recall, precision_at_k=precision,
        hit_rate_at_k=hit_rate, mrr=mrr, ndcg_at_k=ndcg, mean_latency_ms=mean_latency,
        per_question=per_question,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
