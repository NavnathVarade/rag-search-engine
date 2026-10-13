"""CLI with STORE_BACKEND=postgres: ingest writes to the database; a fresh process sees it."""
from __future__ import annotations

from typer.testing import CliRunner

from rag_engine.cli.main import app
from rag_engine.config import get_settings


def test_cli_ingest_persists_to_postgres_and_is_idempotent(pg_clean, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no cli_state.json here: all state must come from Postgres
    monkeypatch.setenv("STORE_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", pg_clean)
    get_settings.cache_clear()
    try:
        doc = tmp_path / "pg.md"
        doc.write_text("# PostgreSQL\n\nCREATE INDEX CONCURRENTLY builds an index without blocking writes.")
        runner = CliRunner()
        assert "Ingested 1 documents" in runner.invoke(app, ["ingest", str(doc)]).output
        again = runner.invoke(app, ["ingest", str(doc)]).output
        assert "Ingested 0 documents" in again and "duplicate" in again
        assert not (tmp_path / "data" / "indexes" / "cli_state.json").exists()
        status = runner.invoke(app, ["index", "status"]).output
        assert "documents=1" in status and "chunks=1" in status
        search = runner.invoke(app, ["search", "create index concurrently", "--mode", "bm25", "--top-k", "1"]).output
        assert "PostgreSQL" in search
    finally:
        get_settings.cache_clear()
