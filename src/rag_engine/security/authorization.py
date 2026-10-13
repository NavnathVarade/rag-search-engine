"""Authorization filtering (spec #30, #95 multi-tenancy).

CRITICAL invariant tested in tests/security/test_authorization.py:
authorization must be applied BEFORE candidates are returned to the
retrieval/generation layers -- never retrieve-then-filter. This module's
`filter_authorized` is the single choke point every retrieval path must
call before results reach context selection or the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag_engine.retrieval.models import RetrievalResult


@dataclass
class AuthContext:
    user_id: str | None
    tenant_id: str | None
    group_ids: frozenset[str]
    is_admin: bool = False


def is_authorized(
    document_visibility: str,
    document_tenant_id: str | None,
    document_allowed_users: list[str],
    document_allowed_groups: list[str],
    auth: AuthContext,
) -> bool:
    if auth.is_admin:
        return True
    if document_visibility == "public":
        # Public docs still respect tenant isolation if the document has a
        # tenant_id set (spec #95 cross-tenant leakage prevention).
        return document_tenant_id is None or document_tenant_id == auth.tenant_id
    if document_tenant_id is not None and document_tenant_id != auth.tenant_id:
        return False
    if auth.user_id and auth.user_id in document_allowed_users:
        return True
    if auth.group_ids & set(document_allowed_groups):
        return True
    return document_visibility == "public"


def filter_authorized(
    results: list[RetrievalResult],
    auth: AuthContext,
    permissions_by_chunk_id: dict[str, dict],
) -> list[RetrievalResult]:
    """`permissions_by_chunk_id[chunk_id]` = {"visibility", "tenant_id",
    "allowed_users", "allowed_groups"}; a chunk missing from this dict is
    treated as public/unrestricted (matches DocumentMetadata's default)."""
    filtered = []
    for r in results:
        perms = permissions_by_chunk_id.get(
            r.chunk_id, {"visibility": "public", "tenant_id": None, "allowed_users": [], "allowed_groups": []}
        )
        if is_authorized(
            perms["visibility"], perms["tenant_id"], perms["allowed_users"], perms["allowed_groups"], auth
        ):
            filtered.append(r)
    return filtered


_VISIBILITIES = ("public", "private", "restricted")
_PUBLIC_ACL: dict = {"visibility": "public", "tenant_id": None, "allowed_users": [], "allowed_groups": []}


def build_acl(
    auth: AuthContext,
    visibility: str = "public",
    allowed_users: list[str] | None = None,
    allowed_groups: list[str] | None = None,
    requested_tenant_id: str | None = None,
) -> dict:
    """ACL stored on every chunk of a document at ingest time.

    The tenant is taken from the AUTHENTICATED caller; a non-admin cannot choose another
    tenant (cross-tenant write prevention). The uploader is recorded as `owner` and, for
    non-public documents, is always allowed to read their own upload."""
    if visibility not in _VISIBILITIES:
        raise ValueError(f"visibility must be one of {_VISIBILITIES}")
    tenant = requested_tenant_id if (auth.is_admin and requested_tenant_id is not None) else auth.tenant_id
    users = list(allowed_users or [])
    if visibility != "public" and auth.user_id and auth.user_id not in users:
        users.append(auth.user_id)
    return {
        "visibility": visibility, "tenant_id": tenant, "allowed_users": users,
        "allowed_groups": list(allowed_groups or []), "owner": auth.user_id,
    }


def acl_of(metadata: dict) -> dict:
    """ACL stored in chunk metadata; chunks without one (ingested via the CLI or before
    auth was enabled) are treated as public/unrestricted -- see docs/security.md."""
    return metadata.get("acl") or _PUBLIC_ACL


def can_read(metadata: dict, auth: AuthContext) -> bool:
    acl = acl_of(metadata)
    return is_authorized(acl["visibility"], acl["tenant_id"], acl["allowed_users"], acl["allowed_groups"], auth)


def can_modify(metadata: dict, auth: AuthContext) -> bool:
    """Delete/modify: admins, or the document's recorded owner. Others get the same
    'not found' as a missing document so existence is not revealed."""
    if auth.is_admin:
        return True
    return bool(auth.user_id) and acl_of(metadata).get("owner") == auth.user_id and can_read(metadata, auth)
