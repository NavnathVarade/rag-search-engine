"""The API's index survives a restart when STATE_FILE is set (new app instance = simulated restart)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.config import Settings
from rag_engine.exceptions import ConfigurationError


def _client(state_file: str) -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, state_file=state_file)))


def _search(c: TestClient, q: str) -> list:
    r = c.post("/api/v1/search", json={"query": q, "retrieval_mode": "bm25", "top_k": 3})
    assert r.status_code == 200
    return r.json()["results"]


def test_documents_survive_restart_and_delete_is_persisted(tmp_path):
    state = str(tmp_path / "idx" / "state.json")
    with _client(state) as c1:
        r = c1.post("/api/v1/documents/ingest", params={"content": "# Redis\n\nRedis AOF logs every write.", "source_uri": "redis"})
        doc_id = r.json()["document_id"]
        assert _search(c1, "redis AOF")
    with _client(state) as c2:  # restart
        assert _search(c2, "redis AOF"), "document lost across restart"
        assert c2.get(f"/api/v1/documents/{doc_id}").status_code == 200
        assert c2.delete(f"/api/v1/documents/{doc_id}").status_code == 200
    with _client(state) as c3:  # restart again: the delete must have been persisted
        assert _search(c3, "redis AOF") == []


def test_without_state_file_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with _client("") as c:
        c.post("/api/v1/documents/ingest", params={"content": "# A\n\nhello world", "source_uri": "a"})
    assert list(tmp_path.iterdir()) == []


def test_corrupt_state_file_fails_loudly(tmp_path):
    bad = tmp_path / "state.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigurationError), _client(str(bad)):
        pass
