"""Dense (embedding-based) retrieval: query -> embedding -> ANN search ->
ranked RetrievalResult list (spec #13)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_engine.embeddings.base import EmbeddingProvider
from rag_engine.retrieval.ann.base import AnnIndex
from rag_engine.retrieval.models import RetrievalMethod, RetrievalResult


@dataclass
class ChunkRecord:
    """Minimal in-memory chunk record used by the dense/BM25 retrievers in
    this reference implementation. The storage layer (storage/repositories)
    maps persisted Postgres rows to/from this shape."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]


class DenseRetriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        ann_index: AnnIndex,
        chunk_store: dict[str, ChunkRecord],
    ) -> None:
        self.embedder = embedder
        self.ann_index = ann_index
        self.chunk_store = chunk_store

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        query_vector = self.embedder.embed_query(query)
        hits = self.ann_index.search(query_vector, top_k=top_k)

        results: list[RetrievalResult] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            record = self.chunk_store.get(chunk_id)
            if record is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=record.document_id,
                    content=record.content,
                    score=score,
                    rank=rank,
                    method=RetrievalMethod.dense,
                    metadata=record.metadata,
                    dense_rank=rank,
                    dense_score=score,
                    explanation=[f"Cosine similarity {score:.3f} to query embedding"],
                )
            )
        return results
