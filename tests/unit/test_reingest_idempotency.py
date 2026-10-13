"""Re-ingesting identical content must not create a second copy -- via the CLI (state file) and
across an API restart (STATE_FILE). Regression for: `rag ingest <dir>` run twice doubled the index."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from rag_engine.api.app import create_app
from rag_engine.cli.main import app as cli_app
from rag_engine.config import Settings


def test_cli_reingest_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    doc = tmp_path / "k8s.md"
    doc.write_text("# Kubernetes\n\nKubernetes restarts a failed container using the Pod restartPolicy.")
    runner = CliRunner()
    first = runner.invoke(cli_app, ["ingest", str(doc)])
    second = runner.invoke(cli_app, ["ingest", str(doc)])
    assert "Ingested 1 documents" in first.output
    assert "Ingested 0 documents" in second.output and "duplicate" in second.output
    status = runner.invoke(cli_app, ["index", "status"])
    assert "documents=1" in status.output and "chunks=1" in status.output


def test_api_duplicate_detection_survives_restart(tmp_path):
    state = str(Path(tmp_path) / "state.json")
    params = {"content": "# Redis\n\nRedis AOF logs every write.", "source_uri": "redis"}
    with TestClient(create_app(Settings(_env_file=None, state_file=state))) as c1:
        first = c1.post("/api/v1/documents/ingest", params=params).json()
    with TestClient(create_app(Settings(_env_file=None, state_file=state))) as c2:  # restart
        again = c2.post("/api/v1/documents/ingest", params=params).json()
    assert first["was_duplicate"] is False
    assert again["was_duplicate"] is True and again["document_id"] == first["document_id"]
