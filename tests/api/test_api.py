"""End-to-end API tests via FastAPI's TestClient (real ASGI app, real
routing/middleware/exception handling -- spec #52 API tests, #39-41)."""
import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_endpoint_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_endpoint_reflects_index_state(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["retriever_available"] is True


def test_ingest_then_search_finds_document(client):
    ingest_resp = client.post(
        "/api/v1/documents/ingest",
        params={"content": "# Kubernetes Restart\n\nKubernetes restarts a failed container using restartPolicy.", "source_uri": "test-doc"},
    )
    assert ingest_resp.status_code == 200
    ingest_body = ingest_resp.json()
    assert ingest_body["chunk_count"] >= 1
    assert not ingest_body["was_duplicate"]

    search_resp = client.post("/api/v1/search", json={"query": "kubernetes restart failed container", "retrieval_mode": "bm25", "top_k": 5})
    assert search_resp.status_code == 200
    search_body = search_resp.json()
    assert len(search_body["results"]) >= 1
    assert "kubernetes" in search_body["results"][0]["content"].lower()
    assert search_body["timings"]["total_ms"] >= 0


def test_search_response_includes_explanation(client):
    client.post("/api/v1/documents/ingest", params={"content": "# Postgres\n\nCREATE INDEX CONCURRENTLY avoids locking.", "source_uri": "pg-doc"})
    resp = client.post("/api/v1/search", json={"query": "CREATE INDEX CONCURRENTLY", "retrieval_mode": "hybrid", "top_k": 5})
    body = resp.json()
    assert body["results"]
    assert body["results"][0]["explanation"]


def test_chat_endpoint_returns_answer_with_citations(client):
    content = "# FastAPI DI\n\nFastAPI dependency injection uses Depends() to declare reusable dependencies."
    client.post("/api/v1/documents/ingest", params={"content": content, "source_uri": "fastapi-doc"})
    resp = client.post("/api/v1/chat", json={"question": "How does FastAPI dependency injection work?", "retrieval_mode": "hybrid", "top_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)
    assert "model" in body


def test_chat_insufficient_evidence_for_unrelated_question(client):
    resp = client.post("/api/v1/chat", json={"question": "What is the airspeed velocity of an unladen swallow?", "retrieval_mode": "hybrid", "top_k": 10})
    body = resp.json()
    assert not body["had_sufficient_evidence"] or "don't have enough evidence" in body["answer"].lower()


def test_delete_document_removes_it_from_index(client):
    ingest_resp = client.post("/api/v1/documents/ingest", params={"content": "# Temp Doc\n\nTemporary content to delete.", "source_uri": "temp-doc"})
    doc_id = ingest_resp.json()["document_id"]

    get_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert get_resp.status_code == 200

    delete_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert delete_resp.status_code == 200

    get_after = client.get(f"/api/v1/documents/{doc_id}")
    assert get_after.status_code == 404
    assert get_after.json()["error"] == "not_found"


def test_search_validates_request_body():
    from rag_engine.api.app import create_app
    with TestClient(create_app()) as c:
        resp = c.post("/api/v1/search", json={"query": "", "top_k": 5})
        assert resp.status_code == 422  # min_length=1 violated


def test_invalid_retrieval_mode_rejected(client):
    resp = client.post("/api/v1/search", json={"query": "test", "retrieval_mode": "not_a_real_mode"})
    assert resp.status_code == 422


def test_error_response_never_leaks_stack_trace(client):
    resp = client.get("/api/v1/documents/nonexistent-doc-id-xyz")
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body
    assert "Traceback" not in str(body)


def test_openapi_docs_available(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/api/v1/search" in schema["paths"]
    assert "/api/v1/chat" in schema["paths"]


def test_tfidf_mode_returns_results_not_silently_empty(client):
    """Regression: the TF-IDF retriever existed but was never wired into the engine, so
    retrieval_mode=tfidf was accepted and always returned zero results."""
    client.post("/api/v1/documents/ingest", params={"content": "# Kubernetes\n\nKubernetes restarts a failed container using the restartPolicy.", "source_uri": "k8s"})
    resp = client.post("/api/v1/search", json={"query": "restart failed container", "retrieval_mode": "tfidf", "top_k": 3})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results and results[0]["method"] == "tfidf"
