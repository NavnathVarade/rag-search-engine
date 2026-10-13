"""Pydantic schemas for document ingestion/management endpoints (spec #39)."""
from __future__ import annotations

from pydantic import BaseModel


class IngestUrlRequest(BaseModel):
    url: str
    visibility: str = "public"


class IngestRepoRequest(BaseModel):
    repo_url: str
    branch: str = "main"


class DocumentResponse(BaseModel):
    document_id: str
    title: str | None
    source_type: str
    source_uri: str
    chunk_count: int
    content_hash: str
    created_at: str


class IngestResponse(BaseModel):
    document_id: str
    was_duplicate: bool
    chunk_count: int
    stage_timings_ms: dict[str, float]
    source_uri: str | None = None
    title: str | None = None
