"""End-to-end test of the retrieval core (embedding -> ANN, tokenizer ->
BM25, RRF fusion) wired together via HybridRetriever, on a small realistic
technical-docs toy corpus. No mocks of the algorithms under test
(spec #154)."""
import pytest

from rag_engine.config import RetrievalMode
from rag_engine.embeddings.local import HashingEmbedder
from rag_engine.retrieval.ann.brute_force import BruteForceIndex
from rag_engine.retrieval.dense import ChunkRecord, DenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.sparse.bm25 import Bm25Scorer
from rag_engine.retrieval.sparse.inverted_index import InvertedIndex
from rag_engine.retrieval.sparse_retriever import Bm25Retriever

DOCS = {
    "k8s-1": (
        "doc-k8s",
        "Kubernetes restarts a failed container according to the Pod's "
        "restartPolicy. The kubelet on the node observes container exits "
        "and, depending on restartPolicy (Always, OnFailure, Never), "
        "restarts the container with an exponential backoff delay.",
    ),
    "k8s-2": (
        "doc-k8s",
        "Readiness probes determine whether a container is ready to accept "
        "traffic; liveness probes determine whether a container should be "
        "restarted. A failing liveness probe triggers a container restart, "
        "while a failing readiness probe only removes the Pod from Service "
        "endpoints without restarting it.",
    ),
    "pg-1": (
        "doc-postgres",
        "CREATE INDEX CONCURRENTLY builds a PostgreSQL index without "
        "holding a lock that blocks writes to the table, at the cost of a "
        "slower build and the requirement that it cannot run inside a "
        "transaction block.",
    ),
    "pg-2": (
        "doc-postgres",
        "PostgreSQL B-tree indexes support equality and range queries and "
        "are the default index type, while GIN indexes are optimized for "
        "composite values such as arrays, jsonb, and full text search.",
    ),
    "fastapi-1": (
        "doc-fastapi",
        "FastAPI dependency injection uses the Depends() function to "
        "declare reusable dependencies that are resolved per-request, "
        "supporting sub-dependencies, caching within a request, and "
        "override for testing.",
    ),
}


@pytest.fixture
def hybrid_retriever() -> HybridRetriever:
    chunk_store = {
        cid: ChunkRecord(chunk_id=cid, document_id=doc_id, content=text, metadata={"source": doc_id})
        for cid, (doc_id, text) in DOCS.items()
    }

    embedder = HashingEmbedder(dimension=256)
    ann = BruteForceIndex(dimension=256)
    vectors = embedder.embed_documents([text for _, text in DOCS.values()])
    ann.add(list(DOCS.keys()), vectors)
    dense = DenseRetriever(embedder=embedder, ann_index=ann, chunk_store=chunk_store)

    index = InvertedIndex()
    for cid, (_, text) in DOCS.items():
        index.add_document(cid, text)
    bm25 = Bm25Retriever(Bm25Scorer(index), chunk_store=chunk_store)

    return HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, rrf_k=60)


def test_bm25_only_finds_exact_technical_term(hybrid_retriever):
    resp = hybrid_retriever.search("CREATE INDEX CONCURRENTLY", mode=RetrievalMode.bm25, top_k=5)
    assert resp.results[0].chunk_id == "pg-1"
    assert resp.results[0].method.value == "bm25"


def test_dense_only_returns_results_for_semantic_query(hybrid_retriever):
    resp = hybrid_retriever.search("how does container restart work", mode=RetrievalMode.dense, top_k=5)
    assert len(resp.results) > 0
    assert resp.results[0].method.value == "dense"


def test_hybrid_mode_surfaces_kubernetes_restart_docs_first(hybrid_retriever):
    resp = hybrid_retriever.search(
        "How does Kubernetes restart a failed container?", mode=RetrievalMode.hybrid, top_k=5
    )
    top_ids = [r.chunk_id for r in resp.results[:2]]
    assert "k8s-1" in top_ids
    # explanation must be populated and mention RRF/rank information
    assert resp.results[0].explanation
    assert any("rank" in line for line in resp.results[0].explanation)


def test_hybrid_response_includes_timings(hybrid_retriever):
    resp = hybrid_retriever.search("fastapi dependency injection", mode=RetrievalMode.hybrid, top_k=5)
    assert resp.timings.total_ms >= 0
    assert resp.timings.dense_ms >= 0
    assert resp.timings.bm25_ms >= 0
    assert resp.timings.fusion_ms >= 0


def test_readiness_vs_liveness_distinguished_by_bm25(hybrid_retriever):
    resp = hybrid_retriever.search(
        "difference between readiness and liveness probes", mode=RetrievalMode.hybrid, top_k=5
    )
    assert resp.results[0].chunk_id == "k8s-2"


def test_dense_retriever_unavailable_degrades_to_bm25_only():
    chunk_store = {
        cid: ChunkRecord(chunk_id=cid, document_id=doc_id, content=text, metadata={})
        for cid, (doc_id, text) in DOCS.items()
    }
    index = InvertedIndex()
    for cid, (_, text) in DOCS.items():
        index.add_document(cid, text)
    bm25 = Bm25Retriever(Bm25Scorer(index), chunk_store=chunk_store)

    retriever = HybridRetriever(dense_retriever=None, bm25_retriever=bm25)
    resp = retriever.search("postgres index", mode=RetrievalMode.hybrid, top_k=5)
    assert "dense_unavailable_falling_back_to_bm25_only" in resp.degraded
    assert len(resp.results) > 0


def test_weighted_hybrid_mode_runs(hybrid_retriever):
    resp = hybrid_retriever.search(
        "postgresql index types", mode=RetrievalMode.hybrid_weighted, top_k=5
    )
    assert len(resp.results) > 0
