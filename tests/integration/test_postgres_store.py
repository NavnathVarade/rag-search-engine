"""PostgresChunkStore against a real PostgreSQL: round trips, atomicity, soft delete."""
from __future__ import annotations

import pytest

from rag_engine.chunking.factory import create_chunker
from rag_engine.config import Settings
from rag_engine.exceptions import PersistenceError
from rag_engine.ingestion.parsers.plaintext import MarkdownParser
from rag_engine.ingestion.pipeline import IngestionPipeline
from rag_engine.retrieval.dense import ChunkRecord
from rag_engine.storage.db import create_engine_from_settings, create_session_factory
from rag_engine.storage.postgres_store import PostgresChunkStore

ACL = {"visibility": "private", "tenant_id": "t1", "allowed_users": ["alice"], "allowed_groups": ["eng"], "owner": "alice"}


def _ingest(text: str, uri: str):
    pipeline = IngestionPipeline(parser=MarkdownParser(), chunker=create_chunker("recursive", max_tokens=200, overlap_tokens=20),
                                 chunking_strategy_name="recursive", embedding_model="hashing", embedding_dimension=384)
    result = pipeline.ingest(text.encode(), source_type="inline", source_uri=uri)
    records = [ChunkRecord(chunk_id=c.chunk_id, document_id=c.document_id, content=c.content,
                           metadata={"source": uri, "acl": ACL, "content_hash": result.metadata.content_hash, **c.metadata}) for c in result.chunks]
    return result, records


@pytest.fixture
async def store(pg_clean):
    engine = create_engine_from_settings(Settings(_env_file=None, database_url=pg_clean))
    yield PostgresChunkStore(create_session_factory(engine))
    await engine.dispose()


async def test_roundtrip_preserves_content_acl_and_hash(store):
    result, records = _ingest("# Redis\n\nRedis AOF logs every write.", "redis.md")
    await store.add_document(result, records)
    loaded = await store.load_chunks()
    assert [c.chunk_id for c in loaded] == [r.chunk_id for r in records]
    assert loaded[0].content == records[0].content
    assert loaded[0].metadata["acl"] == ACL and loaded[0].metadata["source"] == "redis.md"
    assert loaded[0].metadata["content_hash"] == result.metadata.content_hash
    assert await store.ping() is True


async def test_delete_hides_document_and_readding_revives_it(store):
    result, records = _ingest("# A\n\nalpha beta gamma", "a.md")
    await store.add_document(result, records)
    await store.delete_document(result.document_id)
    assert await store.load_chunks() == []
    await store.add_document(result, records)  # same document id, previously soft-deleted
    assert len(await store.load_chunks()) == len(records)


async def test_failed_write_is_atomic(store):
    result, records = _ingest("# B\n\nbeta text here", "b.md")
    result.chunks = [*result.chunks, result.chunks[0]]  # duplicate chunk primary key -> IntegrityError after the document row was written
    with pytest.raises(PersistenceError):
        await store.add_document(result, records)
    assert await store.load_chunks() == [], "a failed write must leave nothing behind (no orphan document row)"


async def test_unreachable_database_raises_persistence_error():
    engine = create_engine_from_settings(Settings(_env_file=None, database_url="postgresql+asyncpg://rag:rag@127.0.0.1:1/none"))
    bad = PostgresChunkStore(create_session_factory(engine))
    assert await bad.ping() is False
    with pytest.raises(PersistenceError):
        await bad.load_chunks()
    await engine.dispose()
