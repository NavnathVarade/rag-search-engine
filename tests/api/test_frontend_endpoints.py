"""API surface added for the web UI: document listing, file upload, URL ingest, system info,
benchmarks, LLM provider selection and static frontend serving. Real app, real pipeline."""
import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.api.engine import build_llm
from rag_engine.config import Settings


@pytest.fixture
def client(tmp_path):
    # frontend_dir is pinned so these tests do not depend on whether frontend/dist has been built
    with TestClient(create_app(Settings(frontend_dir=str(tmp_path / "no-ui")))) as c:
        yield c


def _ingest(client, text, name):
    r = client.post("/api/v1/documents/ingest", params={"content": text, "source_uri": name})
    assert r.status_code == 200, r.text
    return r.json()


def test_list_documents_shows_ingested_with_counts(client):
    d = _ingest(client, "# Redis\n\nRedis persistence uses RDB snapshots and the AOF log.", "redis.md")
    body = client.get("/api/v1/documents").json()
    row = next(x for x in body["documents"] if x["document_id"] == d["document_id"])
    assert row["source"] == "redis.md" and row["chunk_count"] >= 1 and row["can_delete"] is True
    assert body["total"] == len(body["documents"])


def test_upload_markdown_file_then_find_it(client):
    files = {"file": ("nginx.md", io.BytesIO(b"# Nginx\n\nUse proxy_pass to forward requests to an upstream server."), "text/markdown")}
    r = client.post("/api/v1/documents/upload", files=files, data={"visibility": "public"})
    assert r.status_code == 200 and r.json()["source_uri"] == "nginx.md"
    hits = client.post("/api/v1/search", json={"query": "proxy_pass upstream", "retrieval_mode": "bm25", "top_k": 3}).json()["results"]
    assert hits and "proxy_pass" in hits[0]["content"]
    same = io.BytesIO(b"# Nginx\n\nUse proxy_pass to forward requests to an upstream server.")
    again = client.post("/api/v1/documents/upload", files={"file": ("copy.md", same, "text/markdown")})
    assert again.json()["was_duplicate"] is True


def test_upload_rejects_unsupported_type_and_path_in_name(client):
    r = client.post("/api/v1/documents/upload", files={"file": ("a.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert r.status_code == 422 and r.json()["error"] == "unsupported_document_type"
    notes = io.BytesIO(b"plain notes about kafka partitions and consumer groups")
    ok = client.post("/api/v1/documents/upload", files={"file": ("../../etc/notes.txt", notes, "text/plain")})
    assert ok.status_code == 200 and ok.json()["source_uri"] == "notes.txt"


def test_upload_enforces_size_limit():
    app = create_app(Settings(max_document_size_mb=1))
    with TestClient(app) as c:
        r = c.post("/api/v1/documents/upload", files={"file": ("big.txt", io.BytesIO(b"a " * 700_000), "text/plain")})
        assert r.status_code == 413 and r.json()["error"] == "document_too_large"


def test_ingest_url_blocks_private_addresses_by_default(client):
    r = client.post("/api/v1/documents/ingest-url", json={"url": "http://127.0.0.1:9/x"})
    assert r.status_code == 400 and r.json()["error"] == "ssrf_blocked"
    assert client.post("/api/v1/documents/ingest-url", json={"url": "file:///etc/passwd"}).status_code == 422


class _Page(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b"<html><head><title>Celery Guide</title></head><body><h1>Celery</h1><p>Celery workers consume tasks from a broker such as RabbitMQ.</p></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_ingest_url_fetches_html_when_operator_allows_private():
    srv = HTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with TestClient(create_app(Settings(url_ingest_allow_private=True))) as c:
            r = c.post("/api/v1/documents/ingest-url", json={"url": f"http://127.0.0.1:{srv.server_port}/guide"})
            assert r.status_code == 200, r.text
            assert r.json()["title"] == "Celery Guide"
            hits = c.post("/api/v1/search", json={"query": "celery workers broker", "retrieval_mode": "bm25", "top_k": 3}).json()["results"]
            assert hits and "rabbitmq" in hits[0]["content"].lower()
    finally:
        srv.shutdown()


def test_ingest_url_reports_upstream_failure():
    srv = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)  # base handler answers 501
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with TestClient(create_app(Settings(url_ingest_allow_private=True))) as c:
            r = c.post("/api/v1/documents/ingest-url", json={"url": f"http://127.0.0.1:{srv.server_port}/"})
            assert r.status_code == 502 and r.json()["error"] == "fetch_failed"
    finally:
        srv.shutdown()


def test_get_document_returns_chunk_text(client):
    d = _ingest(client, "# Docker\n\nDocker volumes persist data outside the container filesystem.", "docker.md")
    body = client.get(f"/api/v1/documents/{d['document_id']}").json()
    assert body["chunks"] and "volumes" in body["chunks"][0]["content"]


def test_info_reports_live_state_and_stub_llm(client):
    _ingest(client, "# Vault\n\nVault seals protect secrets at rest.", "vault.md")
    body = client.get("/api/v1/info").json()
    assert body["index"]["documents"] >= 1 and body["index"]["chunks"] >= 1
    assert body["config"]["llm_configured"] is False and body["caller"]["is_admin"] is True
    assert body["config"]["embedding_provider"] == "hashing"


def test_benchmarks_lists_reports(tmp_path):
    (tmp_path / "demo.json").write_text('{"dataset": "demo", "results": {"bm25": {"ndcg@10": 0.5}}}')
    with TestClient(create_app(Settings(benchmarks_dir=str(tmp_path)))) as c:
        reports = c.get("/api/v1/benchmarks").json()["reports"]
        assert reports[0]["dataset"] == "demo" and reports[0]["file"] == "demo.json"
    with TestClient(create_app(Settings(benchmarks_dir=str(tmp_path / "missing")))) as c:
        assert c.get("/api/v1/benchmarks").json() == {"reports": []}


def test_llm_provider_selection():
    from rag_engine.generation.local import NullLLMProvider, OllamaProvider
    from rag_engine.generation.openai import OpenAICompatibleProvider

    assert isinstance(build_llm(Settings(llm_provider="openai_compatible", llm_api_key="")), NullLLMProvider)
    assert isinstance(build_llm(Settings(llm_provider="none", llm_api_key="k")), NullLLMProvider)
    assert isinstance(build_llm(Settings(llm_provider="openai_compatible", llm_api_key="k")), OpenAICompatibleProvider)
    assert isinstance(build_llm(Settings(llm_provider="local", llm_model="llama3", llm_base_url="http://localhost:11434/v1")), OllamaProvider)


def test_frontend_is_served_with_spa_fallback_and_api_keeps_priority(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html><body>app-shell</body></html>")
    (tmp_path / "assets" / "a.js").write_text("console.log(1)")
    with TestClient(create_app(Settings(frontend_dir=str(tmp_path)))) as c:
        assert "app-shell" in c.get("/").text
        assert "app-shell" in c.get("/documents").text  # client-side route survives a reload
        assert c.get("/assets/a.js").status_code == 200
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/api/v1/nope").status_code == 404 and c.get("/api/v1/nope").json()["error"] == "not_found"
        assert c.get("/../../etc/passwd").status_code in (200, 404) and "root:" not in c.get("/..%2f..%2fetc/passwd").text


def test_api_unchanged_without_frontend_dir(client):
    assert client.get("/").status_code == 404
