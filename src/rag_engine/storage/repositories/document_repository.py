"""Async repository for documents/chunks (spec #117 repository pattern --
business logic never embeds SQL directly; it calls this repository).
"""
from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.chunking.base import Chunk as ChunkModel
from rag_engine.exceptions import NotFoundError
from rag_engine.ingestion.metadata import DocumentMetadata
from rag_engine.storage.models import Chunk, Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_by_content_hash(self, content_hash: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.content_hash == content_hash, Document.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def upsert_document(self, metadata: DocumentMetadata) -> Document:
        existing = await self.session.get(Document, metadata.document_id)
        if existing:
            existing.title = metadata.title
            existing.deleted_at = None  # re-ingesting a soft-deleted document revives it
            existing.content_hash = metadata.content_hash
            existing.current_version += 1
            await self.session.flush()
            return existing

        doc = Document(
            id=metadata.document_id,
            source_type=metadata.source_type,
            source_uri=metadata.source_uri,
            title=metadata.title,
            author=metadata.author,
            repository=metadata.repository,
            branch=metadata.branch,
            commit_sha=metadata.commit_sha,
            url=metadata.url,
            file_path=metadata.file_path,
            language=metadata.language,
            mime_type=metadata.mime_type,
            content_hash=metadata.content_hash,
            acl_tenant_id=metadata.acl_tenant_id,
            acl_visibility=metadata.acl_visibility,
            doc_metadata=metadata.metadata,
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def add_chunks(self, chunks: list[ChunkModel], metadata_by_chunk_id: dict[str, dict] | None = None) -> None:
        """`metadata_by_chunk_id` overrides a chunk's stored metadata (the API adds source + ACL to it)."""
        overrides = metadata_by_chunk_id or {}
        for c in chunks:
            self.session.add(
                Chunk(
                    id=c.chunk_id,
                    document_id=c.document_id,
                    parent_chunk_id=c.parent_chunk_id,
                    content=c.content,
                    token_count=c.token_count,
                    character_count=c.character_count,
                    position=c.position,
                    section=c.section,
                    heading_path=c.heading_path,
                    language=c.language,
                    chunk_metadata=overrides.get(c.chunk_id, c.metadata),
                )
            )
        await self.session.flush()

    async def get_document(self, document_id: str) -> Document:
        doc = await self.session.get(Document, document_id)
        if doc is None or doc.deleted_at is not None:
            raise NotFoundError(f"document {document_id} not found")
        return doc

    async def soft_delete_document(self, document_id: str) -> None:
        """Deleting a document must remove/invalidate its chunks (spec #91):
        soft-delete on documents, hard-delete on chunks (chunks carry no
        independent value once the parent document is gone, and hard
        deletion guarantees they can never be retrieved again)."""
        from datetime import UTC, datetime

        from sqlalchemy import delete

        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        # Use an ORM-level attribute set (not a bare Core UPDATE) so any
        # copy of this row already in the session's identity map -- e.g.
        # one fetched earlier in the same unit of work -- is updated too.
        # A Core-style bulk UPDATE bypasses the identity map by default
        # and a subsequent session.get() would silently return stale data.
        doc = await self.session.get(Document, document_id)
        if doc is not None:
            doc.deleted_at = datetime.now(UTC)
        else:
            await self.session.execute(
                update(Document).where(Document.id == document_id).values(deleted_at=datetime.now(UTC))
            )
        await self.session.flush()

    async def list_chunks_for_document(self, document_id: str) -> list[Chunk]:
        result = await self.session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.position)
        )
        return list(result.scalars().all())

    async def count_documents(self) -> int:
        from sqlalchemy import func

        result = await self.session.execute(select(func.count()).select_from(Document).where(Document.deleted_at.is_(None)))
        return result.scalar_one()
