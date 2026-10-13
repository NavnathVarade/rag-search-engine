"""Integration tests against a REAL Postgres instance (spec #48, #52
integration tests, #91 deletion, #59 idempotency). No mocking of the
database -- this exercises actual SQL."""
import pytest

from rag_engine.chunking.base import Chunk as ChunkModel
from rag_engine.exceptions import NotFoundError
from rag_engine.ingestion.metadata import DocumentMetadata, new_document_id, now_iso
from rag_engine.storage.repositories.document_repository import DocumentRepository

pytestmark = pytest.mark.integration


def make_metadata(doc_id: str | None = None, content_hash: str = "hash123") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id or new_document_id(),
        source_type="local_file",
        source_uri="docs/k8s.md",
        title="Kubernetes Restart Policy",
        content_hash=content_hash,
        version_hash=content_hash,
        parser_version="v1",
        chunking_strategy="recursive",
        embedding_model="hashing-embedder",
        embedding_dimension=256,
        ingestion_timestamp=now_iso(),
    )


@pytest.mark.asyncio
async def test_upsert_and_retrieve_document(db_session):
    repo = DocumentRepository(db_session)
    metadata = make_metadata()
    doc = await repo.upsert_document(metadata)
    await db_session.commit()

    fetched = await repo.get_document(doc.id)
    assert fetched.title == "Kubernetes Restart Policy"
    assert fetched.content_hash == "hash123"


@pytest.mark.asyncio
async def test_find_by_content_hash_enables_dedup(db_session):
    repo = DocumentRepository(db_session)
    metadata = make_metadata(content_hash="unique-hash-abc")
    await repo.upsert_document(metadata)
    await db_session.commit()

    found = await repo.find_by_content_hash("unique-hash-abc")
    assert found is not None
    not_found = await repo.find_by_content_hash("does-not-exist")
    assert not_found is None


@pytest.mark.asyncio
async def test_add_and_list_chunks(db_session):
    repo = DocumentRepository(db_session)
    metadata = make_metadata()
    doc = await repo.upsert_document(metadata)

    chunks = [
        ChunkModel(chunk_id=f"{doc.id}:0", document_id=doc.id, content="First chunk.", position=0, token_count=2, character_count=12),
        ChunkModel(chunk_id=f"{doc.id}:1", document_id=doc.id, content="Second chunk.", position=1, token_count=2, character_count=13),
    ]
    await repo.add_chunks(chunks)
    await db_session.commit()

    fetched = await repo.list_chunks_for_document(doc.id)
    assert len(fetched) == 2
    assert fetched[0].position == 0
    assert fetched[0].content == "First chunk."


@pytest.mark.asyncio
async def test_get_nonexistent_document_raises_not_found(db_session):
    repo = DocumentRepository(db_session)
    with pytest.raises(NotFoundError):
        await repo.get_document("does-not-exist-at-all")


@pytest.mark.asyncio
async def test_soft_delete_removes_document_and_hard_deletes_chunks(db_session):
    """spec #91: no deleted document should remain retrievable, and its
    chunks must be actually gone (not merely hidden), since chunks are
    also what feeds the retrieval index."""
    repo = DocumentRepository(db_session)
    metadata = make_metadata()
    doc = await repo.upsert_document(metadata)
    chunks = [ChunkModel(chunk_id=f"{doc.id}:0", document_id=doc.id, content="secret content", position=0, token_count=2, character_count=14)]
    await repo.add_chunks(chunks)
    await db_session.commit()

    await repo.soft_delete_document(doc.id)
    await db_session.commit()

    with pytest.raises(NotFoundError):
        await repo.get_document(doc.id)

    remaining_chunks = await repo.list_chunks_for_document(doc.id)
    assert remaining_chunks == []  # hard-deleted, not just filtered


@pytest.mark.asyncio
async def test_count_documents_excludes_deleted(db_session):
    repo = DocumentRepository(db_session)
    doc1 = await repo.upsert_document(make_metadata(content_hash="h1"))
    await repo.upsert_document(make_metadata(content_hash="h2"))
    await db_session.commit()

    count_before = await repo.count_documents()
    assert count_before >= 2

    await repo.soft_delete_document(doc1.id)
    await db_session.commit()

    count_after = await repo.count_documents()
    assert count_after == count_before - 1
