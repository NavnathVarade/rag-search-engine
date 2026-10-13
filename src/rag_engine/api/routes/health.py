"""Health/readiness endpoints (spec #39, #62)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from rag_engine.api.dependencies import get_engine_state
from rag_engine.api.engine import EngineState

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: process is up and able to respond. No dependency checks."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, engine: EngineState = Depends(get_engine_state)) -> JSONResponse:
    """Readiness: required dependencies (index built, embedder available)
    are actually usable, per spec #62."""
    checks = {
        "index_built": engine.is_built,
        "embedder_available": engine.embedder is not None,
        "retriever_available": engine.hybrid_retriever is not None,
    }
    store = request.app.state.store
    if store is not None:
        checks["database"] = await store.ping()
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    return JSONResponse(status_code=status_code, content={"status": "ready" if all_ready else "not_ready", "checks": checks})
