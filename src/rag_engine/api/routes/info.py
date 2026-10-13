"""System information for the web UI: who the caller is, how the server is configured, index size,
and the committed benchmark results. Nothing here is estimated -- every value is read from live state
or from the benchmark JSON files written by scripts/run_beir.py."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from rag_engine import __version__ as rag_version
from rag_engine.api.dependencies import get_auth_context, get_engine_state
from rag_engine.api.engine import EngineState
from rag_engine.generation.local import NullLLMProvider
from rag_engine.security.authorization import AuthContext, can_read

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/info")
async def info(request: Request, engine: EngineState = Depends(get_engine_state), auth: AuthContext = Depends(get_auth_context)) -> dict:
    s = engine.settings
    visible_docs: set[str] = set()
    visible_chunks = 0
    for c in engine.chunk_store.values():
        if auth.is_admin or can_read(c.metadata, auth):
            visible_docs.add(c.document_id)
            visible_chunks += 1
    return {
        "version": rag_version,
        "caller": {"user_id": auth.user_id, "tenant_id": auth.tenant_id, "groups": sorted(auth.group_ids), "is_admin": auth.is_admin},
        "index": {
            "built": engine.is_built, "documents": len(visible_docs), "chunks": visible_chunks,
            "total_documents": engine.document_count, "total_chunks": len(engine.chunk_store),
            "scope": "all documents" if auth.is_admin else "documents you can read",
        },
        "config": {
            "environment": s.environment.value if hasattr(s.environment, "value") else str(s.environment),
            "auth_enabled": s.auth_enabled, "store_backend": s.store_backend, "state_file": bool(s.state_file),
            "embedding_provider": s.embedding_provider, "embedding_model": engine.embedder.model_name(), "embedding_dimension": engine.embedder.dimension(),
            "bm25_stemmer": s.bm25_stemmer, "reranker_enabled": s.reranker_enabled, "min_dense_similarity": s.min_dense_similarity,
            "chunking_strategy": s.chunking_strategy, "chunk_max_tokens": s.chunk_max_tokens,
            "llm_configured": not isinstance(engine.llm, NullLLMProvider), "llm_provider": s.llm_provider, "llm_model": s.llm_model,
            "max_document_size_mb": s.max_document_size_mb, "url_ingest_allow_private": s.url_ingest_allow_private,
            "metrics_enabled": s.metrics_enabled, "metrics_requires_token": bool(s.metrics_token),
        },
    }


@router.get("/benchmarks")
async def benchmarks(request: Request, _auth: AuthContext = Depends(get_auth_context)) -> dict:
    """Benchmark reports found in BENCHMARKS_DIR (written by scripts/run_beir.py); empty if none exist."""
    folder = Path(request.app.state.settings.benchmarks_dir)
    reports = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                reports.append({"file": path.name, **json.loads(path.read_text(encoding="utf-8"))})
            except (OSError, ValueError):
                continue
    return {"reports": reports}
