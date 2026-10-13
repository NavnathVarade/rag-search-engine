"""Authorization-aware retrieval shared by /search and /chat.

Permission filtering happens HERE, immediately after retrieval and before reranking,
context selection or generation -- chunks the caller may not read never reach the
reranker, the context window, the LLM prompt, citations, or the response.

Trade-off (documented in docs/security.md): the ANN/BM25 indexes are global, so
we over-fetch and then filter instead of searching a per-caller index. Admins skip the
filter. If a caller can read only a small fraction of the corpus they may receive fewer
than top_k results.
"""
from __future__ import annotations

from dataclasses import replace

from rag_engine.api.engine import EngineState
from rag_engine.config import RetrievalMode
from rag_engine.observability.metrics import Metrics
from rag_engine.retrieval.hybrid import HybridSearchResponse
from rag_engine.security.authorization import AuthContext, can_read

_OVERFETCH = 5


def retrieve_for(
    engine: EngineState, auth: AuthContext, query: str, mode: RetrievalMode, top_k: int, metrics: Metrics | None = None
) -> HybridSearchResponse:
    response = _retrieve(engine, auth, query, mode, top_k)
    if metrics is not None:
        metrics.observe_retrieval(mode.value, response.timings)
    return response


def _retrieve(engine: EngineState, auth: AuthContext, query: str, mode: RetrievalMode, top_k: int) -> HybridSearchResponse:
    assert engine.hybrid_retriever is not None
    if auth.is_admin:
        return engine.hybrid_retriever.search(query, mode=mode, top_k=top_k)

    fetch_k = max(top_k, min(len(engine.chunk_store), top_k * _OVERFETCH))
    response = engine.hybrid_retriever.search(query, mode=mode, top_k=fetch_k)
    allowed = [
        r for r in response.results
        if (chunk := engine.chunk_store.get(r.chunk_id)) is not None and can_read(chunk.metadata, auth)
    ]
    response.results = [replace(r, rank=i + 1) for i, r in enumerate(allowed[:top_k])]
    return response
