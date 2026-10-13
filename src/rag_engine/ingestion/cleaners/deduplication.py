"""Content hashing for exact-duplicate detection (spec #50, #92). Uses
SHA-256 over the normalized text so whitespace-only differences (which
clean_text already normalizes) don't create spurious "different" hashes."""
from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class DuplicateTracker:
    """In-memory tracker used by the ingestion pipeline to skip re-indexing
    byte-identical content (spec #59 idempotency, #50 deduplication). The
    storage layer's repository backs this with a persistent unique
    constraint on content_hash for the real (multi-process) case."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}  # content_hash -> document_id

    def is_duplicate(self, text: str) -> str | None:
        return self._seen.get(content_hash(text))

    def register(self, text: str, document_id: str) -> None:
        self._seen[content_hash(text)] = document_id

    def register_hash(self, hash_value: str, document_id: str) -> None:
        """Re-register a known content hash (used to restore the tracker after a restart)."""
        self._seen[hash_value] = document_id

    def forget_document(self, document_id: str) -> None:
        """Drop entries for a deleted document so the same content can be ingested again."""
        self._seen = {h: d for h, d in self._seen.items() if d != document_id}
