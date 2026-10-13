"""API with STORE_BACKEND=postgres against a real PostgreSQL: a new app instance is a restart."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.config import Settings
from rag_engine.exceptions import PersistenceError
from rag_engine.security.authentication import generate_api_key


def _client(url: str, **extra) -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, store_backend="postgres", database_url=url, **extra)))


def _hits(c: TestClient, q: str, headers=None) -> list:
    r = c.post("/api/v1/search", json={"query": q, "retrieval_mode": "bm25", "top_k": 3}, headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()["results"]


def test_documents_survive_restart_via_postgres_and_delete_persists(pg_clean):
    params = {"content": "# Redis\n\nRedis AOF logs every write.", "source_uri": "redis"}
    with _client(pg_clean) as c1:
        first = c1.post("/api/v1/documents/ingest", params=params).json()
        assert _hits(c1, "redis AOF")
    with _client(pg_clean) as c2:  # restart: index rebuilt from the database
        assert _hits(c2, "redis AOF")
        again = c2.post("/api/v1/documents/ingest", params=params).json()
        assert again["was_duplicate"] is True and again["document_id"] == first["document_id"]
        assert c2.delete(f"/api/v1/documents/{first['document_id']}").status_code == 200
    with _client(pg_clean) as c3:
        assert _hits(c3, "redis AOF") == []


def test_tenant_acl_survives_restart(pg_clean):
    alice, bob = generate_api_key(), generate_api_key()
    keys = json.dumps({alice[1]: {"user_id": "alice", "tenant_id": "t1"}, bob[1]: {"user_id": "bob", "tenant_id": "t2"}})
    ha, hb = {"x-api-key": alice[0]}, {"x-api-key": bob[0]}
    with _client(pg_clean, auth_enabled=True, api_keys_json=keys) as c1:
        r = c1.post("/api/v1/documents/ingest", headers=ha, params={"content": "# Payroll\n\nconfidential payroll roadmap", "source_uri": "p", "visibility": "private"})
        assert r.status_code == 200
    with _client(pg_clean, auth_enabled=True, api_keys_json=keys) as c2:
        assert len(_hits(c2, "payroll roadmap", ha)) == 1, "owner must still see the document"
        assert _hits(c2, "payroll roadmap", hb) == [], "another tenant must not see it after a restart"


def test_database_is_written_before_memory_and_failed_write_leaves_index_unchanged(pg_clean):
    params = {"content": "# Redis\n\nRedis AOF logs every write.", "source_uri": "redis"}
    with _client(pg_clean) as c:
        async def boom(*a, **k):
            raise PersistenceError("simulated outage")

        real = c.app.state.store.add_document
        c.app.state.store.add_document = boom
        r = c.post("/api/v1/documents/ingest", params=params)
        assert r.status_code == 503 and r.json()["error"] == "persistence_failed"
        assert _hits(c, "redis AOF") == [], "memory must not contain a document the database lacks"
        c.app.state.store.add_document = real
        retry = c.post("/api/v1/documents/ingest", params=params)  # same content must be retryable
        assert retry.status_code == 200 and retry.json()["was_duplicate"] is False
        assert _hits(c, "redis AOF")


def test_ready_reports_database_and_rebuild_reloads_from_it(pg_clean):
    with _client(pg_clean) as c:
        assert c.get("/ready").json()["checks"]["database"] is True
        c.post("/api/v1/documents/ingest", params={"content": "# A\n\nalpha beta", "source_uri": "a"})
        c.app.state.engine.rebuild_from_chunks([])  # simulate a lost/corrupt in-memory index
        assert _hits(c, "alpha beta") == []
        assert c.post("/api/v1/index/rebuild").json()["chunk_count"] == 1
        assert _hits(c, "alpha beta")


def test_startup_fails_loudly_when_database_unreachable():
    with pytest.raises(PersistenceError), _client("postgresql+asyncpg://rag:rag@127.0.0.1:1/none"):
        pass
