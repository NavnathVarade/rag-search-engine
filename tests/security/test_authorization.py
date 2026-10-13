"""Authorization filtering tests (spec #30, #95 -- retrieval must never
expose unauthorized chunks, and cross-tenant leakage must be impossible)."""
from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult
from rag_engine.security.authorization import AuthContext, filter_authorized, is_authorized


def make_result(chunk_id):
    return RetrievalResult(chunk_id=chunk_id, document_id="d", content="c", score=1.0, rank=1, method=RetrievalMethod.hybrid_rrf)


def test_public_document_no_tenant_accessible_to_anyone():
    auth = AuthContext(user_id="u1", tenant_id="tenant-a", group_ids=frozenset())
    assert is_authorized("public", None, [], [], auth)


def test_private_document_requires_explicit_allow():
    auth = AuthContext(user_id="u1", tenant_id="tenant-a", group_ids=frozenset())
    assert not is_authorized("private", "tenant-a", ["u2"], [], auth)
    assert is_authorized("private", "tenant-a", ["u1"], [], auth)


def test_cross_tenant_document_never_accessible():
    """Critical security invariant: a user in tenant-a must never see a
    document belonging to tenant-b, even if visibility=public."""
    auth = AuthContext(user_id="u1", tenant_id="tenant-a", group_ids=frozenset())
    assert not is_authorized("public", "tenant-b", [], [], auth)
    assert not is_authorized("private", "tenant-b", ["u1"], [], auth)  # even if explicitly listed!


def test_group_membership_grants_access():
    auth = AuthContext(user_id="u1", tenant_id="tenant-a", group_ids=frozenset({"eng-team"}))
    assert is_authorized("private", "tenant-a", [], ["eng-team"], auth)
    assert not is_authorized("private", "tenant-a", [], ["other-team"], auth)


def test_admin_bypasses_all_restrictions():
    auth = AuthContext(user_id="admin1", tenant_id="tenant-x", group_ids=frozenset(), is_admin=True)
    assert is_authorized("private", "tenant-a", ["someone-else"], [], auth)


def test_filter_authorized_removes_unauthorized_chunks_before_returning():
    """This is the critical spec #30 invariant: filtering happens on the
    result list itself, so an unauthorized chunk is never present in what
    gets returned to context selection / the LLM."""
    results = [make_result("public-chunk"), make_result("tenant-b-chunk"), make_result("private-chunk")]
    permissions = {
        "public-chunk": {"visibility": "public", "tenant_id": None, "allowed_users": [], "allowed_groups": []},
        "tenant-b-chunk": {"visibility": "public", "tenant_id": "tenant-b", "allowed_users": [], "allowed_groups": []},
        "private-chunk": {"visibility": "private", "tenant_id": "tenant-a", "allowed_users": ["someone-else"], "allowed_groups": []},
    }
    auth = AuthContext(user_id="u1", tenant_id="tenant-a", group_ids=frozenset())

    filtered = filter_authorized(results, auth, permissions)
    filtered_ids = {r.chunk_id for r in filtered}

    assert filtered_ids == {"public-chunk"}
    assert "tenant-b-chunk" not in filtered_ids
    assert "private-chunk" not in filtered_ids


def test_chunk_missing_from_permissions_map_defaults_to_public():
    results = [make_result("legacy-chunk")]
    auth = AuthContext(user_id="u1", tenant_id="tenant-a", group_ids=frozenset())
    filtered = filter_authorized(results, auth, permissions_by_chunk_id={})
    assert len(filtered) == 1
