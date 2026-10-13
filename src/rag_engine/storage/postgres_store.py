"""Durable chunk store on PostgreSQL.

Write path (API/CLI): the database is written FIRST, in one transaction, and the in-memory index is
updated only if that succeeds -- so memory never contains a document the database does not. On startup
(and on `POST /index/rebuild`) every non-deleted chunk is read back and the BM25/ANN indexes are rebuilt
from it; the indexes are derived data. Each chunk's metadata (source, ACL, content hash) is stored in
the chunk's JSON column, and the document's tenant/visibility in the document row, so access control
survives a restart.
"""
from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_engine.exceptions import PersistenceError
from rag_engine.ingestion.pipeline import IngestionResult
from rag_engine.retrieval.dense import ChunkRecord
from rag_engine.storage.models import Chunk, Document, DocumentVersion
from rag_engine.storage.repositories.document_repository import DocumentRepository


class PostgresChunkStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_chunks(self) -> list[ChunkRecord]:
        try:
            async with self._session_factory() as session:
                rows = await session.execute(
                    select(Chunk)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(Document.deleted_at.is_(None))
                    .order_by(Document.created_at, Document.id, Chunk.position)
                )
                return [
                    ChunkRecord(chunk_id=c.id, document_id=c.document_id, content=c.content, metadata=dict(c.chunk_metadata or {}))
                    for c in rows.scalars().all()
                ]
        except Exception as exc:
            raise PersistenceError(f"could not load chunks from the database: {type(exc).__name__}: {exc}") from exc

    async def add_document(self, result: IngestionResult, records: list[ChunkRecord]) -> None:
        """Persist one ingested document and its chunks atomically (all or nothing)."""
        md = result.metadata
        acl = (records[0].metadata.get("acl") if records else None) or {}
        md.acl_tenant_id = acl.get("tenant_id")
        md.acl_visibility = acl.get("visibility", "public")
        md.metadata = {**md.metadata, "acl": acl}
        try:
            async with self._session_factory() as session, session.begin():
                repo = DocumentRepository(session)
                doc = await repo.upsert_document(md)
                session.add(
                    DocumentVersion(
                        document_id=doc.id, version_number=doc.current_version, content_hash=md.content_hash,
                        parser_version=md.parser_version, chunking_strategy=md.chunking_strategy,
                        embedding_model=md.embedding_model, embedding_dimension=md.embedding_dimension,
                    )
                )
                await repo.add_chunks(result.chunks, metadata_by_chunk_id={r.chunk_id: r.metadata for r in records})
        except Exception as exc:
            raise PersistenceError(f"could not store document {md.document_id}: {type(exc).__name__}: {exc}") from exc

    async def delete_document(self, document_id: str) -> None:
        try:
            async with self._session_factory() as session, session.begin():
                await DocumentRepository(session).soft_delete_document(document_id)
        except Exception as exc:
            raise PersistenceError(f"could not delete document {document_id}: {type(exc).__name__}: {exc}") from exc

    async def ping(self) -> bool:
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
