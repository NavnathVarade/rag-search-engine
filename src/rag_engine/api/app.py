"""FastAPI application factory (spec #39, #46 error handling, #62 health,
#96 OpenAPI docs)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_engine.api.engine import build_engine_state
from rag_engine.api.middleware import RequestContextMiddleware
from rag_engine.api.ratelimit import SlidingWindowLimiter
from rag_engine.api.routes import chat, documents, evaluation, health, info, search
from rag_engine.api.routes import metrics as metrics_routes
from rag_engine.config import Settings, get_settings
from rag_engine.exceptions import RagEngineError
from rag_engine.logging import configure_logging
from rag_engine.observability.metrics import Metrics
from rag_engine.security.authentication import Authenticator
from rag_engine.storage.db import create_engine_from_settings, create_session_factory
from rag_engine.storage.postgres_store import PostgresChunkStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    engine = build_engine_state(settings)
    app.state.engine = engine
    db_engine = None
    if settings.store_backend == "postgres":
        # Postgres is the source of truth: rebuild the in-memory indexes from it. Fail startup loudly
        # (rather than serving an empty index) if the database is unreachable or not migrated.
        db_engine = create_engine_from_settings(settings)
        store = PostgresChunkStore(create_session_factory(db_engine))
        engine.rebuild_from_chunks(await store.load_chunks())
        engine.seed_trackers()
        app.state.store = store
    try:
        yield
    finally:
        if db_engine is not None:
            await db_engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="RAG Search Engine",
        description="Hybrid RAG search engine with transparent, from-scratch retrieval algorithms.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.store = None  # set in lifespan when STORE_BACKEND=postgres
    # Raises ConfigurationError at startup if AUTH_ENABLED=true without usable credentials (fail closed).
    app.state.authenticator = Authenticator(settings)
    app.state.limiter = SlidingWindowLimiter()
    app.state.metrics = Metrics(lambda: getattr(app.state, "engine", None))

    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(RagEngineError)
    async def handle_engine_error(request: Request, exc: RagEngineError) -> JSONResponse:
        # Structured, non-leaky error responses (spec #46): never a raw
        # traceback to the client, always a stable machine-readable code.
        headers: dict[str, str] | None = None
        if exc.http_status == 401:
            headers = {"WWW-Authenticate": "Bearer"}
        elif exc.http_status == 429 and "retry_after_s" in exc.details:
            headers = {"Retry-After": str(exc.details["retry_after_s"])}
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict(), headers=headers)

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(evaluation.router)
    app.include_router(info.router)
    if settings.metrics_enabled:
        app.include_router(metrics_routes.router)

    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built React app (frontend/dist) at / when it exists. API routes are registered first and
    keep priority; unknown non-API paths fall back to index.html so client-side routes survive a reload."""
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dist = Path(settings.frontend_dir)
    index = dist / "index.html"
    if not index.is_file():
        return
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="frontend-assets")
    reserved = ("api/", "docs", "redoc", "openapi.json", "health", "ready", "metrics")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        if path.startswith(reserved):
            from rag_engine.exceptions import NotFoundError

            raise NotFoundError("no such endpoint")
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


app = create_app()
