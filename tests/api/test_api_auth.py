"""API-level authentication, authorization and tenant-isolation tests (real ASGI app,
real routing, real retrieval -- nothing about auth or retrieval is mocked)."""
from __future__ import annotations

import json
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from rag_engine.api.app import create_app
from rag_engine.config import Settings
from rag_engine.exceptions import ConfigurationError
from rag_engine.security.authentication import generate_api_key

KEY = "s" * 64
ISS, AUD = "https://issuer.test", "rag-api"
SECRET_A = "Project Zebra launch codes are stored in the vault."


def _tok(sub: str, tenant: str | None, groups=(), roles=()) -> dict:
    claims = {"sub": sub, "iss": ISS, "aud": AUD, "exp": int(time.time()) + 600, "tenant_id": tenant, "groups": list(groups), "roles": list(roles)}
    return {"Authorization": "Bearer " + jwt.encode(claims, KEY, algorithm="HS256")}


ALICE = _tok("alice", "tenantA")  # tenant A
ALICE2 = _tok("alice2", "tenantA", groups=["eng"])  # tenant A, same tenant, group eng
CAROL = _tok("carol", "tenantA", groups=["sales"])
BOB = _tok("bob", "tenantB")  # tenant B
ADMIN = _tok("root", None, roles=["admin"])


@pytest.fixture(scope="module")
def api_key_pair():
    return generate_api_key()


@pytest.fixture
def client(tmp_path, api_key_pair):
    key, digest = api_key_pair
    settings = Settings(
        auth_enabled=True, jwt_key=KEY, jwt_issuer=ISS, jwt_audience=AUD, data_dir=tmp_path,
        api_keys_json=json.dumps({digest: {"user_id": "svc", "tenant_id": "tenantA", "groups": [], "admin": False}}),
        rate_limit_anonymous_per_min=10_000, rate_limit_authenticated_per_min=10_000,  # high: tests below build their own client to test limits
    )
    with TestClient(create_app(settings)) as c:
        yield c


def _ingest(client, headers, content=SECRET_A, **params):
    r = client.post("/api/v1/documents/ingest", params={"content": content, "source_uri": "doc", **params}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _search(client, headers, q="Zebra launch codes vault"):
    r = client.post("/api/v1/search", json={"query": q, "retrieval_mode": "bm25", "top_k": 10}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["results"]


# --- authentication gate ------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/search"), ("post", "/api/v1/chat"), ("post", "/api/v1/documents/ingest"),
        ("get", "/api/v1/documents/x"), ("delete", "/api/v1/documents/x"), ("post", "/api/v1/index/rebuild"),
        ("get", "/api/v1/evaluation/runs"),
    ],
)
def test_every_api_route_requires_credentials(client, method, path):
    resp = getattr(client, method)(path)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json()["error"] == "unauthorized"


def test_health_endpoints_stay_open(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_bad_tokens_get_401(client):
    expired = {"Authorization": "Bearer " + jwt.encode({"sub": "a", "iss": ISS, "aud": AUD, "exp": int(time.time()) - 5}, KEY, algorithm="HS256")}
    forged = {"Authorization": "Bearer " + jwt.encode({"sub": "a", "iss": ISS, "aud": AUD, "exp": int(time.time()) + 60}, "z" * 64, algorithm="HS256")}
    for hdrs in (expired, forged, {"Authorization": "Basic abc"}, {"X-API-Key": "rag_wrong"}):
        assert client.post("/api/v1/search", json={"query": "x"}, headers=hdrs).status_code == 401


def test_api_key_works(client, api_key_pair):
    key, _ = api_key_pair
    assert client.post("/api/v1/search", json={"query": "anything"}, headers={"X-API-Key": key}).status_code == 200


def test_app_refuses_to_start_with_auth_enabled_and_no_credentials():
    with pytest.raises(ConfigurationError):
        create_app(Settings(auth_enabled=True))


# --- tenant isolation -----------------------------------------------------
def test_other_tenant_cannot_find_read_or_delete_a_document(client):
    doc = _ingest(client, ALICE)
    doc_id = doc["document_id"]

    assert [r["document_id"] for r in _search(client, ALICE)] == [doc_id]
    assert _search(client, BOB) == []  # B's search never sees A's chunk
    assert client.get(f"/api/v1/documents/{doc_id}", headers=BOB).status_code == 404  # existence not revealed
    assert client.delete(f"/api/v1/documents/{doc_id}", headers=BOB).status_code == 404
    assert client.get(f"/api/v1/documents/{doc_id}", headers=ALICE).status_code == 200  # still there


def test_chat_never_leaks_other_tenants_text_or_citations(client):
    _ingest(client, ALICE)
    r = client.post("/api/v1/chat", json={"question": "Where are the Zebra launch codes?", "retrieval_mode": "bm25", "top_k": 5}, headers=BOB)
    assert r.status_code == 200
    body = json.dumps(r.json())
    assert "Zebra" not in body.replace("Where are the Zebra launch codes?", "")
    assert r.json()["citations"] == [] and r.json()["retrieved_chunks"] == []
    assert r.json()["had_sufficient_evidence"] is False


def test_same_content_in_two_tenants_is_not_reported_as_duplicate(client):
    a = _ingest(client, ALICE)
    b = _ingest(client, BOB)
    assert b["was_duplicate"] is False and b["document_id"] != a["document_id"]
    assert [r["document_id"] for r in _search(client, BOB)] == [b["document_id"]]


def test_non_admin_cannot_write_into_another_tenant(client):
    doc = _ingest(client, BOB, tenant_id="tenantA")  # the tenant_id parameter must be ignored for non-admins
    assert [r["document_id"] for r in _search(client, BOB)] == [doc["document_id"]]
    assert _search(client, ALICE) == []


def test_private_documents_are_visible_only_to_owner_and_allowed_principals(client):
    doc = _ingest(client, ALICE, visibility="private", allowed_groups=["eng"])
    ids = lambda h: [r["document_id"] for r in _search(client, h)]  # noqa: E731
    assert ids(ALICE) == [doc["document_id"]]  # owner
    assert ids(ALICE2) == [doc["document_id"]]  # allowed via group "eng"
    assert ids(CAROL) == []  # same tenant, wrong group
    assert ids(BOB) == []  # other tenant
    assert ids(ADMIN) == [doc["document_id"]]


def test_only_owner_or_admin_can_delete(client):
    doc = _ingest(client, ALICE)
    assert client.delete(f"/api/v1/documents/{doc['document_id']}", headers=ALICE2).status_code == 404  # same tenant, not owner
    assert client.delete(f"/api/v1/documents/{doc['document_id']}", headers=ALICE).status_code == 200
    assert _search(client, ALICE) == []
    again = _ingest(client, ALICE)  # deleting forgot the dedup entry, so re-ingest works
    assert again["was_duplicate"] is False


def test_search_results_do_not_expose_acl_lists(client):
    _ingest(client, ALICE, visibility="private", allowed_users=["someone-else"])
    results = _search(client, ALICE)
    assert results and "acl" not in results[0]["metadata"]


def test_invalid_visibility_is_rejected(client):
    r = client.post("/api/v1/documents/ingest", params={"content": "x y z", "visibility": "world-readable"}, headers=ALICE)
    assert r.status_code == 422


# --- admin-only routes ------------------------------------------------------
def test_rebuild_and_evaluation_are_admin_only(client):
    for path, method in (("/api/v1/index/rebuild", "post"), ("/api/v1/evaluation/runs", "get")):
        assert getattr(client, method)(path, headers=ALICE).status_code == 403
        assert getattr(client, method)(path, headers=ADMIN).status_code == 200


def test_evaluation_results_dir_is_not_caller_controlled(client):
    r = client.get("/api/v1/evaluation/runs", params={"results_dir": "/etc"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["count"] == 0  # the parameter is ignored


# --- rate limiting (applied after authentication) -----------------------------
def _limited_client(tmp_path, anon=3, authed=5):
    settings = Settings(auth_enabled=True, jwt_key=KEY, jwt_issuer=ISS, jwt_audience=AUD, data_dir=tmp_path,
                        rate_limit_anonymous_per_min=anon, rate_limit_authenticated_per_min=authed)
    return TestClient(create_app(settings))


def test_authenticated_callers_are_limited_per_identity_with_retry_after(tmp_path):
    with _limited_client(tmp_path, authed=5) as c:
        codes = [c.post("/api/v1/search", json={"query": "x"}, headers=ALICE).status_code for _ in range(7)]
        assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]
        limited = c.post("/api/v1/search", json={"query": "x"}, headers=ALICE)
        assert limited.json()["error"] == "rate_limited" and int(limited.headers["retry-after"]) >= 1
        assert c.post("/api/v1/search", json={"query": "x"}, headers=BOB).status_code == 200  # other identity unaffected


def test_rotating_fake_tokens_cannot_bypass_the_limit(tmp_path):
    with _limited_client(tmp_path, anon=3) as c:
        codes = [c.post("/api/v1/search", json={"query": "x"}, headers={"Authorization": f"Bearer fake-{i}"}).status_code for i in range(8)]
        assert codes[:3] == [401, 401, 401] and set(codes[3:]) == {429}
        assert c.post("/api/v1/search", json={"query": "x"}, headers=ALICE).status_code == 200  # real users share no bucket with the attacker's failures


def test_search_route_is_rate_limited_too(tmp_path):
    with _limited_client(tmp_path, authed=2) as c:
        statuses = [c.post("/api/v1/search", json={"query": "x"}, headers=ALICE).status_code for _ in range(3)]
        assert statuses == [200, 200, 429]
