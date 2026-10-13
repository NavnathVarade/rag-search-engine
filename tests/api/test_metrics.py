"""Prometheus /metrics: the exposition text is parsed with prometheus_client's own parser and
values are asserted -- not just a 200."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from rag_engine.api.app import create_app
from rag_engine.config import Settings
from rag_engine.security.authentication import generate_api_key


def _scrape(client, headers=None) -> dict[str, list[tuple[dict, float]]]:
    r = client.get("/metrics", headers=headers)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    out: dict[str, list[tuple[dict, float]]] = {}
    for family in text_string_to_metric_families(r.text):
        for s in family.samples:
            out.setdefault(s.name, []).append((s.labels, s.value))
    return out


def _value(samples, name, **labels) -> float:
    return sum(v for lab, v in samples.get(name, []) if all(lab.get(k) == x for k, x in labels.items()))


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, frontend_dir=str(tmp_path / 'no-ui')))) as c:  # no UI build: unknown paths are 404s
        yield c


def test_http_request_counters_and_latency_histogram(client):
    for _ in range(3):
        client.get("/health")
    s = _scrape(client)
    assert _value(s, "rag_http_requests_total", method="GET", route="/health", status="200") == 3
    assert _value(s, "rag_http_request_duration_seconds_count", method="GET", route="/health") == 3
    assert _value(s, "rag_http_request_duration_seconds_bucket", route="/health", le="+Inf") == 3


def test_route_label_is_the_template_and_cardinality_stays_bounded(client):
    for i in range(25):
        client.get(f"/api/v1/documents/doc_{i}")  # 25 different ids -> 404s
        client.get(f"/no/such/path/{i}")
    s = _scrape(client)
    routes = {lab["route"] for lab, _ in s["rag_http_requests_total"]}
    assert "/api/v1/documents/{document_id}" in routes and "unmatched" in routes
    assert not any("doc_" in r or "/no/such" in r for r in routes)
    assert len(routes) <= 5


def test_ingest_search_chat_delete_update_domain_metrics_and_index_gauges(client):
    ing = client.post("/api/v1/documents/ingest", params={"content": "# Redis\n\nRedis AOF logs every write to disk.", "source_uri": "r"})
    assert ing.status_code == 200
    dup = client.post("/api/v1/documents/ingest", params={"content": "# Redis\n\nRedis AOF logs every write to disk.", "source_uri": "r"})
    assert dup.json()["was_duplicate"] is True
    client.post("/api/v1/search", json={"query": "redis aof", "retrieval_mode": "hybrid", "top_k": 3})
    client.post("/api/v1/search", json={"query": "redis aof", "retrieval_mode": "bm25", "top_k": 3})
    client.post("/api/v1/chat", json={"question": "What does AOF do?", "retrieval_mode": "hybrid", "top_k": 3})

    s = _scrape(client)
    assert _value(s, "rag_ingestion_documents_total", outcome="ingested") == 1
    assert _value(s, "rag_ingestion_documents_total", outcome="duplicate") == 1
    assert _value(s, "rag_ingestion_chunks_total") == 1
    assert _value(s, "rag_retrieval_duration_seconds_count", mode="hybrid") == 2  # one search + one chat
    assert _value(s, "rag_retrieval_duration_seconds_count", mode="bm25") == 1
    assert _value(s, "rag_retrieval_stage_duration_seconds_count", stage="bm25") >= 1
    assert _value(s, "rag_chat_requests_total", sufficient_evidence="true") == 1
    assert _value(s, "rag_index_documents") == 1 and _value(s, "rag_index_chunks") == 1
    assert _value(s, "rag_index_vocabulary_terms") > 0

    client.delete(f"/api/v1/documents/{ing.json()['document_id']}")
    s = _scrape(client)
    assert _value(s, "rag_index_documents") == 0 and _value(s, "rag_index_chunks") == 0  # gauges read live


def test_metrics_never_contain_query_text_or_document_content(client):
    client.post("/api/v1/documents/ingest", params={"content": "top secret zebra plan", "source_uri": "s"})
    client.post("/api/v1/search", json={"query": "zebra", "retrieval_mode": "bm25"})
    body = client.get("/metrics").text
    assert "zebra" not in body and "secret" not in body


def test_auth_failures_and_rate_limits_are_counted(tmp_path):
    key, digest = generate_api_key()
    settings = Settings(
        auth_enabled=True, api_keys_json=json.dumps({digest: {"user_id": "svc"}}), data_dir=tmp_path,
        rate_limit_anonymous_per_min=2, rate_limit_authenticated_per_min=2, metrics_token="scrape-token",
    )
    with TestClient(create_app(settings)) as c:
        for _ in range(4):  # 2 x 401, then throttled 429s
            c.post("/api/v1/search", json={"query": "x"}, headers={"X-API-Key": "wrong"})
        for _ in range(3):  # 2 ok, 1 limited
            c.post("/api/v1/search", json={"query": "x"}, headers={"X-API-Key": key})
        s = _scrape(c, {"Authorization": "Bearer scrape-token"})
    assert _value(s, "rag_auth_failures_total") == 4
    assert _value(s, "rag_rate_limited_total", scope="auth_failure") == 2
    assert _value(s, "rag_rate_limited_total", scope="identity") == 1


def test_metrics_endpoint_token_protection(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, metrics_token="scrape-token"))) as c:
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer scrape-token"}).status_code == 200


def test_metrics_can_be_disabled(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, metrics_enabled=False))) as c:
        assert c.get("/metrics").status_code == 404


def test_two_apps_in_one_process_do_not_share_or_collide(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as a, TestClient(create_app(Settings(data_dir=tmp_path))) as b:
        a.get("/health")
        assert _value(_scrape(a), "rag_http_requests_total", route="/health") == 1
        assert _value(_scrape(b), "rag_http_requests_total", route="/health") == 0


def test_route_label_stays_bounded_when_the_web_ui_is_served(tmp_path):
    """With a UI build present, every unknown path is answered by one catch-all route; the label is its
    template, so the number of series does not grow with the number of distinct URLs."""
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<html></html>")
    with TestClient(create_app(Settings(data_dir=tmp_path, frontend_dir=str(ui)))) as c:
        for i in range(25):
            assert c.get(f"/some/client/route/{i}").status_code == 200
        routes = {lab["route"] for lab, _ in _scrape(c)["rag_http_requests_total"]}
        assert routes == {"/{path:path}"}
