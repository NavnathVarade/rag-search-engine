"""Real neural cross-encoder reranker via sentence-transformers'
`CrossEncoder`. Requires `uv sync --extra embeddings` and network
access to download model weights on first use -- not exercised in this
sandbox (no route to huggingface.co), analogous to
embeddings/local.py:SentenceTransformerEmbedder. Scores each
(query, candidate) pair jointly through the model (true cross-attention),
which is what makes this more accurate than the bi-encoder dense
retriever, at the cost of one model forward pass per candidate.
"""
from __future__ import annotations

from rag_engine.exceptions import RerankingError
from rag_engine.retrieval.models import RetrievalResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu") -> None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:
            raise RerankingError(
                "sentence-transformers is not installed. Install with "
                "`uv sync --extra embeddings` to use the neural reranker."
            ) from exc
        try:
            self._model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise RerankingError(f"failed to load reranker model {model_name}: {exc}") from exc
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []
        pairs = [(query, c.content) for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankingError(f"cross-encoder scoring failed: {exc}") from exc

        scored = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
        reranked = []
        for new_rank, (score, candidate) in enumerate(scored[:top_k], start=1):
            candidate.reranker_score = float(score)
            candidate.explanation = [*candidate.explanation, f"Cross-encoder ({self.model_name}) score {float(score):.3f}"]
            candidate.rank = new_rank
            reranked.append(candidate)
        return reranked
