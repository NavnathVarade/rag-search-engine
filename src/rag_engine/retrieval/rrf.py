"""Reciprocal Rank Fusion (RRF), implemented from scratch.

    RRF(d) = sum_i  weight_i / (k + rank_i(d))

  where rank_i(d) is the 1-based rank of document d in ranker i's result
  list (documents not present in ranker i's list simply do not contribute
  a term for that ranker -- they are not penalized with an infinite rank).

Reference: Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods", SIGIR 2009.

Why RRF instead of "just average the similarity scores":
  BM25 scores and cosine similarities live on completely different,
  un-normalized scales (BM25 can range from 0 to double digits depending
  on corpus statistics; cosine is bounded [-1, 1]) and are not
  probabilistically comparable. Averaging them directly would let whichever
  score happens to have larger magnitude dominate the fusion for reasons
  that have nothing to do with actual relevance. RRF sidesteps this by
  fusing *ranks*, not raw scores -- rank is scale-free and comparable
  across arbitrarily different scoring functions. `k` (default 60, from
  the original paper) dampens the influence of any single very-high rank
  and controls how quickly a ranker's contribution decays with rank.

This module returns not just the fused score but a full explanation
object per chunk (spec #19, #21) showing each ranker's rank/score and its
individual RRF contribution, so fusion is auditable rather than opaque.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult


@dataclass
class RankerContribution:
    ranker: str
    rank: int
    score: float
    rrf_contribution: float


@dataclass
class FusedResult:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict
    rrf_score: float
    contributions: list[RankerContribution]
    final_rank: int = 0

    def reason(self) -> list[str]:
        reasons = []
        rankers = {c.ranker for c in self.contributions}
        if len(rankers) > 1:
            reasons.append(f"Ranked by {len(rankers)} retrievers: {', '.join(sorted(rankers))}")
        for c in sorted(self.contributions, key=lambda x: x.rank):
            reasons.append(f"{c.ranker}: rank {c.rank}, score {c.score:.3f}, RRF contribution {c.rrf_contribution:.5f}")
        if len(rankers) > 1:
            reasons.append("Strong agreement across retrievers boosts confidence")
        return reasons

    def to_retrieval_result(self) -> RetrievalResult:
        dense = next((c for c in self.contributions if c.ranker == "dense"), None)
        bm25 = next((c for c in self.contributions if c.ranker == "bm25"), None)
        return RetrievalResult(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            content=self.content,
            score=self.rrf_score,
            rank=self.final_rank,
            method=RetrievalMethod.hybrid_rrf,
            metadata=self.metadata,
            explanation=self.reason(),
            dense_rank=dense.rank if dense else None,
            dense_score=dense.score if dense else None,
            bm25_rank=bm25.rank if bm25 else None,
            bm25_score=bm25.score if bm25 else None,
            rrf_score=self.rrf_score,
        )


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievalResult]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse multiple named ranked lists of RetrievalResult via RRF.

    `ranked_lists`: e.g. {"dense": [...], "bm25": [...]}, each already
    sorted by that ranker's own score (rank == list position, 1-based).
    `weights`: optional per-ranker weight (weighted RRF, spec #19). Missing
    ranker defaults to weight 1.0.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    weights = weights or {}

    accum: dict[str, dict] = {}  # chunk_id -> {content, document_id, metadata, contributions}

    for ranker_name, results in ranked_lists.items():
        weight = weights.get(ranker_name, 1.0)
        seen_in_this_ranker: set[str] = set()
        for result in results:
            if result.chunk_id in seen_in_this_ranker:
                continue  # de-dup: a ranker should not list the same doc twice
            seen_in_this_ranker.add(result.chunk_id)

            contribution = weight / (k + result.rank)
            entry = accum.setdefault(
                result.chunk_id,
                {
                    "document_id": result.document_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "contributions": [],
                    "rrf_score": 0.0,
                },
            )
            entry["contributions"].append(
                RankerContribution(
                    ranker=ranker_name, rank=result.rank, score=result.score, rrf_contribution=contribution
                )
            )
            entry["rrf_score"] += contribution

    fused = [
        FusedResult(
            chunk_id=chunk_id,
            document_id=data["document_id"],
            content=data["content"],
            metadata=data["metadata"],
            rrf_score=data["rrf_score"],
            contributions=data["contributions"],
        )
        for chunk_id, data in accum.items()
    ]
    fused.sort(key=lambda f: f.rrf_score, reverse=True)
    for rank, f in enumerate(fused, start=1):
        f.final_rank = rank

    if top_k is not None:
        fused = fused[:top_k]
    return fused
