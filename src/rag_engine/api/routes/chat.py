"""Chat/RAG endpoint (spec #41): retrieve -> rerank -> select context ->
generate -> validate citations, fully explainable."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from rag_engine.api.dependencies import get_auth_context, get_engine_state
from rag_engine.api.engine import EngineState
from rag_engine.api.retrieval_access import retrieve_for
from rag_engine.api.routes.search import _to_result_item
from rag_engine.api.schemas.chat import ChatRequest, ChatResponse, CitationItem
from rag_engine.api.schemas.search import SearchTimings
from rag_engine.security.authorization import AuthContext

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    http_request: Request,
    request: ChatRequest,
    engine: EngineState = Depends(get_engine_state),
    auth: AuthContext = Depends(get_auth_context),
) -> ChatResponse:
    engine.require_built()
    assert engine.context_selector is not None
    assert engine.answerer is not None
    settings = engine.settings

    t_retrieve = time.perf_counter()
    search_response = retrieve_for(engine, auth, request.question, request.retrieval_mode, request.top_k, http_request.app.state.metrics)
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

    candidates = search_response.results
    t_rerank = time.perf_counter()
    if settings.reranker_enabled and candidates:
        candidates = engine.reranker.rerank(request.question, candidates, top_k=settings.rerank_top_k)
    rerank_ms = (time.perf_counter() - t_rerank) * 1000

    selection = engine.context_selector.select(candidates)
    answer = await engine.answerer.answer(request.question, selection)

    metrics = http_request.app.state.metrics
    metrics.chat_requests.labels(sufficient_evidence=str(answer.had_sufficient_evidence).lower()).inc()
    metrics.llm_tokens.labels(kind="prompt").inc(answer.prompt_tokens)
    metrics.llm_tokens.labels(kind="completion").inc(answer.completion_tokens)

    citations = [
        CitationItem(marker=c.marker, chunk_id=c.chunk_id, document_id=c.document_id, source_label=c.source_label)
        for c in answer.citations
    ]

    return ChatResponse(
        question=request.question,
        answer=answer.answer,
        citations=citations,
        retrieved_chunks=[_to_result_item(s.result) for s in selection.selected],
        had_sufficient_evidence=answer.had_sufficient_evidence,
        injection_flags=answer.injection_flags,
        model=answer.model,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        timings=SearchTimings(
            embedding_ms=search_response.timings.embedding_ms, dense_ms=search_response.timings.dense_ms,
            bm25_ms=search_response.timings.bm25_ms, fusion_ms=search_response.timings.fusion_ms,
            reranking_ms=rerank_ms, total_ms=retrieve_ms + rerank_ms,
        ),
    )
