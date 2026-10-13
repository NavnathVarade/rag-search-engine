"""Professional CLI (spec #42) built with Typer. All commands operate on
the same in-process EngineState used by the API (rag_engine.api.engine),
so `rag search` and `POST /api/v1/search` are guaranteed to behave
identically -- there is exactly one retrieval implementation.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from rag_engine.api.engine import EngineState, build_engine_state, tracker_scope
from rag_engine.chunking.factory import available_strategies, create_chunker
from rag_engine.config import RetrievalMode, get_settings
from rag_engine.evaluation.dataset import load_dataset
from rag_engine.evaluation.evaluator import evaluate_retrieval
from rag_engine.evaluation.experiment import run_experiment
from rag_engine.ingestion.cleaners.deduplication import DuplicateTracker
from rag_engine.ingestion.loaders.local_file import discover_files, read_file_with_limit
from rag_engine.ingestion.parsers.factory import get_parser_for_content_type, get_parser_for_file
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.retrieval.dense import ChunkRecord
from rag_engine.storage.db import create_engine_from_settings, create_session_factory
from rag_engine.storage.postgres_store import PostgresChunkStore

app = typer.Typer(help="rag-engine: hybrid RAG search engine CLI")
index_app = typer.Typer(help="Index management")
app.add_typer(index_app, name="index")
console = Console()

_STATE_PATH = Path("data/indexes/cli_state.json")


async def _with_store(fn: Callable[[PostgresChunkStore], Awaitable[Any]]) -> Any:
    """Run `fn(store)` against Postgres with a short-lived engine (the CLI is synchronous)."""
    db_engine = create_engine_from_settings(get_settings())
    try:
        return await fn(PostgresChunkStore(create_session_factory(db_engine)))
    finally:
        await db_engine.dispose()


def _load_or_build_engine() -> EngineState:
    settings = get_settings()
    engine = build_engine_state(settings)
    if settings.store_backend == "postgres":
        engine.rebuild_from_chunks(asyncio.run(_with_store(lambda store: store.load_chunks())))
        engine.seed_trackers()
        return engine
    if _STATE_PATH.exists():
        records = [ChunkRecord(**r) for r in json.loads(_STATE_PATH.read_text())]
        engine.rebuild_from_chunks(records)
        engine.seed_trackers()
    return engine


def _persist_engine(engine: EngineState) -> None:
    if engine.settings.store_backend == "postgres":
        return  # documents were written to Postgres as they were ingested
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"chunk_id": c.chunk_id, "document_id": c.document_id, "content": c.content, "metadata": c.metadata}
        for c in engine.chunk_store.values()
    ]
    _STATE_PATH.write_text(json.dumps(records))


@app.command()
def health() -> None:
    """Check engine health/config sanity."""
    settings = get_settings()
    console.print(f"[green]OK[/green] environment={settings.environment.value} chunking={settings.chunking_strategy}")


auth_app = typer.Typer(help="Authentication helpers")
app.add_typer(auth_app, name="auth")


@auth_app.command("create-key")
def create_key(
    user_id: str = typer.Option(..., help="User/service identity for this key"),
    tenant_id: str = typer.Option(None, help="Tenant this key belongs to"),
    group: list[str] = typer.Option([], "--group", help="Group membership (repeatable)"),
    admin: bool = typer.Option(False, "--admin", help="Grant administrator rights"),
) -> None:
    """Generate an API key. The key is shown ONCE; add the printed entry to API_KEYS_JSON (only the hash is stored)."""
    import json

    from rag_engine.security.authentication import generate_api_key

    key, digest = generate_api_key()
    entry = {digest: {"user_id": user_id, "tenant_id": tenant_id, "groups": list(group), "admin": admin}}
    console.print(f"[bold]API key (store it now, it cannot be recovered):[/bold] {key}")
    console.print("[bold]Merge this entry into API_KEYS_JSON:[/bold]")
    console.print(json.dumps(entry))


@app.command()
def config() -> None:
    """Print the effective configuration."""
    settings = get_settings()
    console.print_json(settings.model_dump_json())


@app.command()
def ingest(
    path: str = typer.Argument(None, help="Local file or directory to ingest"),
    url: str = typer.Option(None, help="URL to ingest"),
    allow_private_network: bool = typer.Option(False, "--allow-private-network", help="Let --url reach private addresses (operator use only; blocked by default)"),
    chunking_strategy: str = typer.Option(None, help=f"One of {available_strategies()}"),
) -> None:
    """Ingest local files, a directory tree, or a URL."""
    settings = get_settings()
    engine = _load_or_build_engine()
    strategy = chunking_strategy or settings.chunking_strategy
    chunker = create_chunker(strategy, max_tokens=settings.chunk_max_tokens, overlap_tokens=settings.chunk_overlap_tokens)

    total_chunks = 0
    total_docs = 0
    new_records: list[ChunkRecord] = []
    pending: list[tuple[Any, list[ChunkRecord]]] = []  # (IngestionResult, its records) to write to Postgres
    tracker = engine.duplicate_trackers.setdefault(tracker_scope(None), DuplicateTracker())

    def _ingest_bytes(content: bytes, source_uri: str, content_type: str | None = None) -> None:
        nonlocal total_chunks, total_docs
        try:
            if content_type is not None:
                parser = get_parser_for_content_type(content_type, source_uri, max_pdf_pages=settings.max_pdf_pages)
            else:
                parser = get_parser_for_file(source_uri, max_pdf_pages=settings.max_pdf_pages)
        except Exception as exc:
            console.print(f"[yellow]skip[/yellow] {source_uri}: {exc}")
            return
        pipeline = IngestionPipeline(
            parser=parser, chunker=chunker, chunking_strategy_name=strategy,
            embedding_model=engine.embedder.model_name(), embedding_dimension=engine.embedder.dimension(), duplicate_tracker=tracker,
        )
        try:
            result = pipeline.ingest(content, source_type="url" if content_type is not None else "local_file", source_uri=source_uri)
        except Exception as exc:
            console.print(f"[red]error[/red] {source_uri}: {exc}")
            return
        if not result.was_duplicate:
            total_docs += 1
            total_chunks += len(result.chunks)
            doc_records = []
            for c in result.chunks:
                meta = {"source": source_uri, "content_hash": result.metadata.content_hash, **c.metadata}
                doc_records.append(ChunkRecord(chunk_id=c.chunk_id, document_id=c.document_id, content=c.content, metadata=meta))
            new_records.extend(doc_records)
            pending.append((result, doc_records))
        suffix = " (duplicate)" if result.was_duplicate else ""
        console.print(f"  {source_uri}: {len(result.chunks)} chunks{suffix}")

    if url:
        from rag_engine.ingestion.loaders.url_loader import UrlLoader

        loader = UrlLoader(max_size_mb=settings.max_document_size_mb, allow_private_hosts=allow_private_network)
        try:
            fetched = loader.fetch_document(url)
        except Exception as exc:
            console.print(f"[red]error[/red] could not fetch {url}: {type(exc).__name__}: {exc}")
            raise typer.Exit(1) from exc
        _ingest_bytes(fetched.content, fetched.final_url, fetched.content_type)
    elif path:
        p = Path(path)
        if p.is_file():
            _ingest_bytes(read_file_with_limit(p, settings.max_document_size_mb), str(p))
        elif p.is_dir():
            files = discover_files(p, max_files=settings.max_files_per_repo)
            for f in files:
                try:
                    _ingest_bytes(read_file_with_limit(f, settings.max_document_size_mb), str(f))
                except Exception as exc:
                    console.print(f"[red]error[/red] {f}: {exc}")
        else:
            console.print(f"[red]path not found: {path}[/red]")
            raise typer.Exit(1)
    else:
        console.print("[red]must provide a path or --url[/red]")
        raise typer.Exit(1)

    if settings.store_backend == "postgres" and pending:
        async def _save(store: PostgresChunkStore) -> None:
            for res, recs in pending:
                await store.add_document(res, recs)

        try:
            asyncio.run(_with_store(_save))  # database first; the local index is rebuilt only after this succeeds
        except Exception as exc:
            console.print(f"[red]error[/red] could not write to Postgres: {exc}")
            raise typer.Exit(1) from exc
    all_records = list(engine.chunk_store.values()) + new_records
    engine.rebuild_from_chunks(all_records)
    _persist_engine(engine)
    console.print(f"[green]Ingested {total_docs} documents, {total_chunks} chunks. Index rebuilt.[/green]")


@index_app.command("build")
def index_build() -> None:
    """(Re)build the retrieval index from currently ingested chunks."""
    engine = _load_or_build_engine()
    engine.rebuild_from_chunks(list(engine.chunk_store.values()))
    console.print(f"[green]Index built: {engine.document_count} documents, {len(engine.chunk_store)} chunks[/green]")


@index_app.command("rebuild")
def index_rebuild() -> None:
    index_build()


@index_app.command("status")
def index_status() -> None:
    engine = _load_or_build_engine()
    vocab = engine.inverted_index.vocabulary_size
    console.print(f"built={engine.is_built} documents={engine.document_count} chunks={len(engine.chunk_store)} vocab={vocab}")


@index_app.command("inspect")
def index_inspect() -> None:
    engine = _load_or_build_engine()
    table = Table(title="Index Inspection")
    table.add_column("field")
    table.add_column("value")
    table.add_row("embedding_model", engine.embedder.model_name())
    table.add_row("embedding_dimension", str(engine.embedder.dimension()))
    table.add_row("document_count", str(engine.document_count))
    table.add_row("chunk_count", str(len(engine.chunk_store)))
    table.add_row("bm25_vocabulary_size", str(engine.inverted_index.vocabulary_size))
    table.add_row("bm25_avg_doc_length", f"{engine.inverted_index.average_document_length:.1f}")
    table.add_row("ann_backend", engine.settings.ann_backend)
    console.print(table)


@index_app.command("validate")
def index_validate() -> None:
    engine = _load_or_build_engine()
    problems = []
    for chunk_id, record in engine.chunk_store.items():
        if not record.content.strip():
            problems.append(f"empty content: {chunk_id}")
    if engine.ann_index and engine.ann_index.size != len(engine.chunk_store):
        problems.append(f"ANN index size ({engine.ann_index.size}) != chunk store size ({len(engine.chunk_store)})")
    if problems:
        console.print(f"[red]{len(problems)} problems found:[/red]")
        for p in problems:
            console.print(f"  - {p}")
        raise typer.Exit(1)
    console.print("[green]Index is valid.[/green]")


@app.command()
def search(
    query: str,
    mode: RetrievalMode = typer.Option(RetrievalMode.hybrid, "--mode"),
    top_k: int = typer.Option(10, "--top-k"),
) -> None:
    """Search the index and print ranked results with explanations."""
    engine = _load_or_build_engine()
    engine.require_built()
    assert engine.hybrid_retriever is not None
    response = engine.hybrid_retriever.search(query, mode=mode, top_k=top_k)

    table = Table(title=f"Search results for: {query!r} (mode={mode.value})")
    table.add_column("rank")
    table.add_column("chunk_id")
    table.add_column("score")
    table.add_column("content")
    for r in response.results:
        table.add_row(str(r.rank), r.chunk_id, f"{r.score:.4f}", r.content[:80])
    console.print(table)
    t = response.timings
    console.print(f"[dim]total_ms={t.total_ms:.1f} dense_ms={t.dense_ms:.1f} bm25_ms={t.bm25_ms:.1f} fusion_ms={t.fusion_ms:.1f}[/dim]")
    if response.degraded:
        console.print(f"[yellow]degraded: {response.degraded}[/yellow]")


@app.command()
def chat(question: str, mode: RetrievalMode = typer.Option(RetrievalMode.hybrid, "--mode")) -> None:
    """Ask a question and get a cited answer."""
    import asyncio

    engine = _load_or_build_engine()
    engine.require_built()
    assert engine.hybrid_retriever is not None
    assert engine.context_selector is not None
    assert engine.answerer is not None
    retriever, selector, answerer = engine.hybrid_retriever, engine.context_selector, engine.answerer

    async def _run() -> None:
        search_response = retriever.search(question, mode=mode, top_k=50)
        candidates = search_response.results
        if engine.settings.reranker_enabled and candidates:
            candidates = engine.reranker.rerank(question, candidates, top_k=engine.settings.rerank_top_k)
        selection = selector.select(candidates)
        answer = await answerer.answer(question, selection)
        console.print(f"\n[bold]Answer:[/bold] {answer.answer}\n")
        console.print(f"[bold]Citations:[/bold] {[c.chunk_id for c in answer.citations]}")
        console.print(f"[dim]sufficient_evidence={answer.had_sufficient_evidence} model={answer.model}[/dim]")

    asyncio.run(_run())


@app.command()
def debug(question: str, mode: RetrievalMode = typer.Option(RetrievalMode.hybrid, "--mode")) -> None:
    """Full retrieval diagnostic report for a query (spec #76, #132)."""
    engine = _load_or_build_engine()
    engine.require_built()
    assert engine.hybrid_retriever is not None

    console.rule(f"QUERY: {question}")
    response = engine.hybrid_retriever.search(question, mode=mode, top_k=10)
    console.print(f"[bold]retrieval_mode:[/bold] {response.retrieval_mode}")
    console.print(f"[bold]timings:[/bold] {response.timings}")
    console.print(f"[bold]degraded:[/bold] {response.degraded or 'none'}\n")

    for r in response.results[:5]:
        console.rule(f"rank {r.rank}: {r.chunk_id}", style="dim")
        console.print(f"score={r.score:.4f} method={r.method.value}")
        for line in r.explanation:
            console.print(f"  - {line}")


@app.command()
def evaluate(
    dataset: str = typer.Option(..., "--dataset"),
    mode: RetrievalMode = typer.Option(RetrievalMode.hybrid, "--mode"),
    top_k: int = typer.Option(50, "--top-k"),
) -> None:
    """Run the retrieval evaluation harness against a JSONL dataset."""
    engine = _load_or_build_engine()
    engine.require_built()
    assert engine.hybrid_retriever is not None
    questions = load_dataset(Path(dataset))
    assert engine.hybrid_retriever is not None
    retriever = engine.hybrid_retriever

    def retrieve_fn(query: str, k: int) -> tuple[list, float]:
        t0 = time.perf_counter()
        response = retriever.search(query, mode=mode, top_k=k)
        return response.results, (time.perf_counter() - t0) * 1000

    report = evaluate_retrieval(questions, retrieve_fn)
    console.print_json(json.dumps(report.to_dict(), indent=2))

    record = run_experiment(
        name=f"eval-{mode.value}",
        config={"mode": mode.value, "top_k": top_k, "bm25_k1": engine.settings.bm25_k1, "bm25_b": engine.settings.bm25_b, "rrf_k": engine.settings.rrf_k},
        metrics=report.to_dict(),
        dataset_path=Path(dataset),
    )
    console.print(f"[green]Experiment recorded: {record.experiment_id}[/green]")


@app.command()
def documents(action: str = typer.Argument("audit")) -> None:
    """Document/index audit tooling (spec #130)."""
    engine = _load_or_build_engine()
    if action != "audit":
        console.print(f"[red]unknown action: {action}[/red]")
        raise typer.Exit(1)
    docs = {c.document_id for c in engine.chunk_store.values()}
    empty = [c.chunk_id for c in engine.chunk_store.values() if not c.content.strip()]
    console.print(f"documents={len(docs)} chunks={len(engine.chunk_store)} empty_chunks={len(empty)}")


if __name__ == "__main__":
    app()
