"""Search endpoint (spec #39-40): the explainable hybrid-retrieval API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from rag_engine.api.dependencies import get_auth_context, get_engine_state
from rag_engine.api.engine import EngineState
from rag_engine.api.retrieval_access import retrieve_for
from rag_engine.api.schemas.search import (
    RankerDetail,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTimings,
)
from rag_engine.retrieval.models import RetrievalResult
from rag_engine.security.authorization import AuthContext

router = APIRouter(prefix="/api/v1", tags=["search"])


def _to_result_item(r: RetrievalResult) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=r.chunk_id, document_id=r.document_id, content=r.content, rank=r.rank, score=r.score,
        method=r.method.value,
        dense=RankerDetail(rank=r.dense_rank, score=r.dense_score) if r.dense_rank is not None else None,
        bm25=RankerDetail(rank=r.bm25_rank, score=r.bm25_score) if r.bm25_rank is not None else None,
        rrf_score=r.rrf_score, reranker_score=r.reranker_score, explanation=r.explanation,
        metadata={k: v for k, v in r.metadata.items() if k != "acl"},  # never expose other users' ACL lists
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    http_request: Request,
    request: SearchRequest,
    engine: EngineState = Depends(get_engine_state),
    auth: AuthContext = Depends(get_auth_context),
) -> SearchResponse:
    engine.require_built()
    response = retrieve_for(engine, auth, request.query, request.retrieval_mode, request.top_k, http_request.app.state.metrics)

    return SearchResponse(
        query=response.query,
        retrieval_mode=response.retrieval_mode,
        results=[_to_result_item(r) for r in response.results],
        timings=SearchTimings(
            embedding_ms=response.timings.embedding_ms, dense_ms=response.timings.dense_ms,
            bm25_ms=response.timings.bm25_ms, fusion_ms=response.timings.fusion_ms,
            total_ms=response.timings.total_ms,
        ),
        degraded=response.degraded,
    )
