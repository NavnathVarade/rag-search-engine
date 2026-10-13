"""Document + chunk metadata model (spec #7)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class DocumentMetadata:
    document_id: str
    source_type: str  # "local_file" | "url" | "git_repo"
    source_uri: str
    title: str | None
    content_hash: str
    version_hash: str
    parser_version: str
    chunking_strategy: str
    embedding_model: str
    embedding_dimension: int
    ingestion_timestamp: str
    author: str | None = None
    repository: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    url: str | None = None
    file_path: str | None = None
    language: str | None = None
    mime_type: str | None = None
    creation_timestamp: str | None = None
    metadata: dict = field(default_factory=dict)
    acl_tenant_id: str | None = None
    acl_visibility: str = "public"  # "public" | "private" | "restricted"
    acl_allowed_users: list[str] = field(default_factory=list)
    acl_allowed_groups: list[str] = field(default_factory=list)
    document_version: int = 1


def new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:16]}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
