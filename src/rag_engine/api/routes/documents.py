"""Document ingestion + management endpoints (spec #39, #91-93).

Every route requires an authenticated caller (see api/dependencies.get_auth_context).
Documents carry an ACL (tenant, visibility, allowed users/groups, owner) on each chunk;
reads, deletes and lookups are checked against it, and a document the caller may not see
is reported as "not found" so its existence is not revealed.
"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from rag_engine.api.dependencies import get_auth_context, get_engine_state, require_admin
from rag_engine.api.engine import EngineState, tracker_scope
from rag_engine.api.schemas.documents import IngestResponse, IngestUrlRequest
from rag_engine.chunking.factory import create_chunker
from rag_engine.exceptions import DocumentTooLargeError, InvalidRequestError, NotFoundError, RagEngineError
from rag_engine.ingestion.cleaners.deduplication import DuplicateTracker
from rag_engine.ingestion.parsers.base import DocumentParser
from rag_engine.ingestion.parsers.factory import get_parser_for_content_type, get_parser_for_file
from rag_engine.ingestion.parsers.plaintext import MarkdownParser
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.retrieval.dense import ChunkRecord
from rag_engine.security.authorization import AuthContext, build_acl, can_modify, can_read

router = APIRouter(prefix="/api/v1", tags=["documents"])

# One duplicate tracker per access scope (see engine.tracker_scope for why they must not be shared).
def _tracker_for(engine: EngineState, acl: dict) -> DuplicateTracker:
    return engine.duplicate_trackers.setdefault(tracker_scope(acl), DuplicateTracker())


def _doc_chunks(engine: EngineState, document_id: str) -> list[ChunkRecord]:
    return [c for c in engine.chunk_store.values() if c.document_id == document_id]


class UpstreamFetchError(RagEngineError):
    code = "fetch_failed"
    http_status = 502


async def _ingest_bytes(
    http_request: Request, engine: EngineState, auth: AuthContext, *, content: bytes, parser: DocumentParser,
    source_type: str, source_uri: str, visibility: str, allowed_users: list[str], allowed_groups: list[str], tenant_id: str | None,
) -> IngestResponse:
    """Shared ingest path for inline text, uploaded files and fetched URLs: parse, clean, dedupe, chunk,
    write to the database first (when configured), then update the in-memory indexes."""
    settings = engine.settings
    if len(content) > settings.max_document_size_mb * 1024 * 1024:
        raise DocumentTooLargeError("document exceeds the size limit")
    try:
        acl = build_acl(auth, visibility, allowed_users, allowed_groups, tenant_id)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    chunker = create_chunker(settings.chunking_strategy, max_tokens=settings.chunk_max_tokens, overlap_tokens=settings.chunk_overlap_tokens)
    pipeline = IngestionPipeline(
        parser=parser, chunker=chunker, chunking_strategy_name=settings.chunking_strategy,
        embedding_model=engine.embedder.model_name(), embedding_dimension=engine.embedder.dimension(),
        duplicate_tracker=_tracker_for(engine, acl),
    )
    result = await run_in_threadpool(pipeline.ingest, content, source_type, source_uri)

    metrics = http_request.app.state.metrics
    metrics.ingest_documents.labels(outcome="duplicate" if result.was_duplicate else "ingested").inc()
    metrics.ingest_chunks.inc(len(result.chunks))
    if not result.was_duplicate and result.chunks:
        new_records = [
            ChunkRecord(
                chunk_id=c.chunk_id, document_id=c.document_id, content=c.content,
                metadata={"source": source_uri, "acl": acl, "content_hash": result.metadata.content_hash, **c.metadata},
            )
            for c in result.chunks
        ]
        store = http_request.app.state.store
        if store is not None:
            try:
                await store.add_document(result, new_records)  # database first; memory only changes if this succeeds
            except Exception:
                _tracker_for(engine, acl).forget_document(result.document_id)  # allow a retry of the same content
                raise
        engine.rebuild_from_chunks(list(engine.chunk_store.values()) + new_records)
        engine.save_state()

    return IngestResponse(
        document_id=result.document_id, was_duplicate=result.was_duplicate,
        chunk_count=len(result.chunks), stage_timings_ms=result.stage_timings_ms,
        source_uri=source_uri, title=result.metadata.title,
    )


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_text_document(
    http_request: Request,
    content: str,
    source_uri: str = "inline",
    visibility: str = "public",
    allowed_users: list[str] = Query(default_factory=list),
    allowed_groups: list[str] = Query(default_factory=list),
    tenant_id: str | None = None,
    engine: EngineState = Depends(get_engine_state),
    auth: AuthContext = Depends(get_auth_context),
) -> IngestResponse:
    """Ingest inline text/markdown. The document's tenant is the caller's tenant (only an
    admin may set `tenant_id`)."""
    return await _ingest_bytes(
        http_request, engine, auth, content=content.encode("utf-8"), parser=MarkdownParser(), source_type="inline",
        source_uri=source_uri, visibility=visibility, allowed_users=allowed_users, allowed_groups=allowed_groups, tenant_id=tenant_id,
    )


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    http_request: Request,
    file: UploadFile = File(...),
    visibility: str = Form("public"),
    engine: EngineState = Depends(get_engine_state),
    auth: AuthContext = Depends(get_auth_context),
) -> IngestResponse:
    """Ingest an uploaded file (.pdf .html .htm .md .txt and common source-code extensions). The parser is
    chosen from the file name; the upload is read with a hard size cap."""
    limit = engine.settings.max_document_size_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise DocumentTooLargeError("document exceeds the size limit")
    name = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1][:200] or "upload"
    parser = get_parser_for_file(name, max_pdf_pages=engine.settings.max_pdf_pages)
    return await _ingest_bytes(
        http_request, engine, auth, content=data, parser=parser, source_type="upload", source_uri=name,
        visibility=visibility, allowed_users=[], allowed_groups=[], tenant_id=None,
    )


@router.post("/documents/ingest-url", response_model=IngestResponse)
async def ingest_url(
    http_request: Request,
    body: IngestUrlRequest,
    engine: EngineState = Depends(get_engine_state),
    auth: AuthContext = Depends(get_auth_context),
) -> IngestResponse:
    """Fetch a web page or PDF and ingest it. Requests are SSRF-checked (private and loopback addresses are
    refused unless the operator sets URL_INGEST_ALLOW_PRIVATE=true) and size-capped."""
    from rag_engine.ingestion.loaders.url_loader import UrlLoader

    if urlparse(body.url).scheme not in ("http", "https"):
        raise InvalidRequestError("url must start with http:// or https://")
    settings = engine.settings
    loader = UrlLoader(max_size_mb=settings.max_document_size_mb, allow_private_hosts=settings.url_ingest_allow_private)
    try:
        fetched = await run_in_threadpool(loader.fetch_document, body.url)
    except RagEngineError:
        raise
    except httpx.HTTPStatusError as exc:
        raise UpstreamFetchError(f"the server answered {exc.response.status_code} for {body.url}") from exc
    except httpx.HTTPError as exc:
        raise UpstreamFetchError(f"could not fetch {body.url}: {type(exc).__name__}") from exc
    parser = get_parser_for_content_type(fetched.content_type, fetched.final_url, max_pdf_pages=settings.max_pdf_pages)
    return await _ingest_bytes(
        http_request, engine, auth, content=fetched.content, parser=parser, source_type="url", source_uri=fetched.final_url,
        visibility=body.visibility, allowed_users=[], allowed_groups=[], tenant_id=None,
    )


@router.get("/documents")
async def list_documents(engine: EngineState = Depends(get_engine_state), auth: AuthContext = Depends(get_auth_context)) -> dict:
    """Documents the caller may read, with chunk counts. Documents they may not read are omitted."""
    docs: dict[str, dict] = {}
    for c in engine.chunk_store.values():
        d = docs.get(c.document_id)
        if d is None:
            if not can_read(c.metadata, auth):
                continue
            acl = c.metadata.get("acl") or {}
            d = docs[c.document_id] = {
                "document_id": c.document_id, "source": c.metadata.get("source"), "title": c.metadata.get("title"),
                "visibility": acl.get("visibility", "public"), "owner": acl.get("owner"),
                "chunk_count": 0, "characters": 0, "can_delete": can_modify(c.metadata, auth),
            }
        d["chunk_count"] += 1
        d["characters"] += len(c.content)
    items = sorted(docs.values(), key=lambda x: (str(x["source"] or ""), x["document_id"]))
    return {"documents": items, "total": len(items)}


@router.delete("/documents/{document_id}")
async def delete_document(
    request: Request, document_id: str, engine: EngineState = Depends(get_engine_state), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    chunks = _doc_chunks(engine, document_id)
    if not chunks or not can_modify(chunks[0].metadata, auth):
        raise NotFoundError(f"document {document_id} not found in the active index")
    if request.app.state.store is not None:
        await request.app.state.store.delete_document(document_id)  # database first
    engine.rebuild_from_chunks([c for c in engine.chunk_store.values() if c.document_id != document_id])
    for tracker in engine.duplicate_trackers.values():
        tracker.forget_document(document_id)
    engine.save_state()
    return {"document_id": document_id, "deleted": True}


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str, engine: EngineState = Depends(get_engine_state), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    chunks = _doc_chunks(engine, document_id)
    if not chunks or not can_read(chunks[0].metadata, auth):
        raise NotFoundError(f"document {document_id} not found in the active index")
    return {
        "document_id": document_id, "chunk_count": len(chunks), "chunk_ids": [c.chunk_id for c in chunks],
        "source": chunks[0].metadata.get("source"), "title": chunks[0].metadata.get("title"),
        "chunks": [{"chunk_id": c.chunk_id, "content": c.content} for c in chunks],
    }


@router.post("/index/rebuild")
async def rebuild_index(request: Request, engine: EngineState = Depends(get_engine_state), _admin: AuthContext = Depends(require_admin)) -> dict:
    """Rebuild the retrieval indexes -- from the database when STORE_BACKEND=postgres (source of truth),
    otherwise from the in-memory chunk records."""
    store = request.app.state.store
    records = await store.load_chunks() if store is not None else list(engine.chunk_store.values())
    engine.rebuild_from_chunks(records)
    if store is not None:
        engine.seed_trackers()
    return {"rebuilt": True, "document_count": engine.document_count, "chunk_count": len(engine.chunk_store)}
