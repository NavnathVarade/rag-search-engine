"""In-process retrieval engine state: wires together the embedding
provider, ANN index, BM25 index, hybrid retriever, reranker, context
selector, and answerer into one object the API/CLI can share.

Indexes are held in memory and rebuilt from chunk records. Where the records are stored depends on
STORE_BACKEND: Postgres (storage/postgres_store.py, the source of truth) or, for local use, memory with an
optional JSON snapshot (`save_state`/`load_state`, STATE_FILE). The ANN/BM25 structures are derived data, so
rebuilding them after a restart is always well-defined.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from rag_engine.config import Settings
from rag_engine.context.selector import ContextSelector
from rag_engine.embeddings.base import EmbeddingProvider
from rag_engine.embeddings.cache import CachedEmbeddingProvider
from rag_engine.embeddings.local import HashingEmbedder
from rag_engine.exceptions import ConfigurationError, IndexNotBuiltError
from rag_engine.generation.answerer import RagAnswerer
from rag_engine.generation.base import LLMProvider
from rag_engine.generation.local import NullLLMProvider
from rag_engine.ingestion.cleaners.deduplication import DuplicateTracker
from rag_engine.reranking.base import Reranker
from rag_engine.reranking.scoring import HeuristicReranker
from rag_engine.retrieval.ann.base import AnnIndex
from rag_engine.retrieval.ann.brute_force import BruteForceIndex
from rag_engine.retrieval.ann.hnsw import HNSWIndex
from rag_engine.retrieval.dense import ChunkRecord, DenseRetriever
from rag_engine.retrieval.hybrid import HybridRetriever
from rag_engine.retrieval.sparse.bm25 import Bm25Scorer
from rag_engine.retrieval.sparse.inverted_index import InvertedIndex
from rag_engine.retrieval.sparse.tfidf import TfidfScorer
from rag_engine.retrieval.sparse.tokenizer import Tokenizer, TokenizerConfig
from rag_engine.retrieval.sparse_retriever import Bm25Retriever, TfidfRetriever

logger = structlog.get_logger("engine")


def tracker_scope(acl: dict | None) -> str:
    """Duplicate detection is scoped per access scope: one shared tracker would tell tenant B
    "duplicate of doc X" for content only tenant A has (a cross-tenant existence leak)."""
    if not acl:
        return "local"
    return f"{acl['tenant_id']}|" + ("public" if acl["visibility"] == "public" else f"owner:{acl.get('owner')}")


@dataclass
class EngineState:
    settings: Settings
    embedder: EmbeddingProvider
    llm: LLMProvider
    reranker: Reranker
    chunk_store: dict[str, ChunkRecord] = field(default_factory=dict)
    inverted_index: InvertedIndex = field(default_factory=InvertedIndex)
    ann_index: AnnIndex | None = None
    hybrid_retriever: HybridRetriever | None = None
    context_selector: ContextSelector | None = None
    answerer: RagAnswerer | None = None
    is_built: bool = False
    # Duplicate-content trackers, one per access scope (see api/routes/documents.py). Held on the engine,
    # not in module globals, so they share the index's lifetime and tests/apps stay isolated.
    duplicate_trackers: dict[str, DuplicateTracker] = field(default_factory=dict)
    document_count: int = 0

    def require_built(self) -> None:
        if not self.is_built:
            raise IndexNotBuiltError("index has not been built yet -- run `rag index build` first")

    def seed_trackers(self) -> None:
        """Restore duplicate trackers from stored chunks (each carries its document's content hash),
        so re-ingesting identical content after a restart is still recognised as a duplicate."""
        for record in self.chunk_store.values():
            digest = record.metadata.get("content_hash")
            if digest:
                self.duplicate_trackers.setdefault(tracker_scope(record.metadata.get("acl")), DuplicateTracker()).register_hash(digest, record.document_id)

    def save_state(self) -> None:
        """Atomically snapshot all chunks to STATE_FILE (no-op when unset). The BM25/ANN indexes are
        derived data and are rebuilt from these records on load."""
        if not self.settings.state_file:
            return
        path = Path(self.settings.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [{"chunk_id": c.chunk_id, "document_id": c.document_id, "content": c.content, "metadata": c.metadata} for c in self.chunk_store.values()]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(records))
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a half-written snapshot

    def load_state(self) -> int:
        """Rebuild the index from STATE_FILE if it exists. Returns the number of chunks loaded."""
        if not self.settings.state_file or not Path(self.settings.state_file).exists():
            return 0
        try:
            records = [ChunkRecord(**r) for r in json.loads(Path(self.settings.state_file).read_text())]
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(f"STATE_FILE {self.settings.state_file} is unreadable or corrupt: {exc}") from exc
        self.rebuild_from_chunks(records)
        self.seed_trackers()
        return len(records)

    def rebuild_from_chunks(self, chunk_records: list[ChunkRecord]) -> None:
        self.chunk_store = {c.chunk_id: c for c in chunk_records}
        stemmer = self.settings.bm25_stemmer
        tokenizer = Tokenizer(TokenizerConfig(stem=stemmer != "none", stemmer=stemmer if stemmer != "none" else "light"))
        self.inverted_index = InvertedIndex(tokenizer=tokenizer)
        for c in chunk_records:
            self.inverted_index.add_document(c.chunk_id, c.content)

        dim = self.embedder.dimension()
        if self.settings.ann_backend == "hnsw":
            self.ann_index = HNSWIndex(dimension=dim, m=self.settings.hnsw_m, ef_construction=self.settings.hnsw_ef_construction, ef_search=self.settings.hnsw_ef_search)
        else:
            self.ann_index = BruteForceIndex(dimension=dim)

        if chunk_records:
            vectors = self.embedder.embed_documents([c.content for c in chunk_records])
            self.ann_index.add([c.chunk_id for c in chunk_records], vectors)

        dense = DenseRetriever(self.embedder, self.ann_index, self.chunk_store) if chunk_records else None
        bm25_scorer = Bm25Scorer(self.inverted_index, k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        bm25 = Bm25Retriever(bm25_scorer, self.chunk_store)

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=dense, bm25_retriever=bm25, tfidf_retriever=TfidfRetriever(TfidfScorer(self.inverted_index), self.chunk_store), rrf_k=self.settings.rrf_k,
            dense_weight=self.settings.dense_weight, sparse_weight=self.settings.sparse_weight,
        )
        self.context_selector = ContextSelector(
            max_context_tokens=self.settings.max_context_tokens,
            max_chunks_per_document=self.settings.max_chunks_per_document,
            dedup_threshold=self.settings.context_similarity_dedup_threshold,
            min_dense_similarity=self.settings.min_dense_similarity,
        )
        self.answerer = RagAnswerer(self.llm, temperature=self.settings.llm_temperature, max_tokens=self.settings.llm_max_tokens)
        self.document_count = len({c.document_id for c in chunk_records})
        self.is_built = True


_OPENAI_PROVIDERS = {"openai", "openai_compatible"}
_ST_PROVIDERS = {"sentence_transformers", "sentence-transformers"}


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Choose the embedding backend from EMBEDDING_PROVIDER. An unknown value fails at startup instead of
    silently falling back to the hashing embedder (which has no semantic understanding)."""
    provider = settings.embedding_provider.lower()
    if provider == "hashing":
        return HashingEmbedder(dimension=settings.embedding_dimension)
    if provider in _ST_PROVIDERS:
        from rag_engine.embeddings.local import SentenceTransformerEmbedder  # heavy import: only when selected

        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model, device=settings.embedding_device, batch_size=settings.embedding_batch_size,
        )
    raise ConfigurationError(f"unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}: use 'hashing' or 'sentence_transformers'")


def build_llm(settings: Settings) -> LLMProvider:
    """Choose the generation backend from settings. Without credentials the offline stub is used
    (its answers say so, and a warning is logged); it is never silently substituted for a provider
    that is configured, and an unknown LLM_PROVIDER fails at startup."""
    from rag_engine.generation.local import OllamaProvider
    from rag_engine.generation.openai import OpenAICompatibleProvider

    provider = settings.llm_provider.lower()
    if provider == "none":
        return NullLLMProvider()
    if provider == "local":
        return OllamaProvider(model=settings.llm_model, base_url=settings.llm_base_url, timeout_s=settings.llm_timeout_s)
    if provider in _OPENAI_PROVIDERS:
        if not settings.llm_api_key:
            logger.warning("llm_api_key_missing", provider=provider, detail="LLM_API_KEY is empty: answers use the offline stub")
            return NullLLMProvider()
        return OpenAICompatibleProvider(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model,
            timeout_s=settings.llm_timeout_s, max_retries=settings.llm_max_retries,
        )
    raise ConfigurationError(f"unknown LLM_PROVIDER {settings.llm_provider!r}: use 'openai' (or 'openai_compatible'), 'local' or 'none'")


def build_engine_state(settings: Settings) -> EngineState:
    base_embedder = build_embedder(settings)
    embedder = CachedEmbeddingProvider(base_embedder, max_size=settings.embedding_cache_size)
    llm = build_llm(settings)
    reranker: Reranker = HeuristicReranker()
    state = EngineState(settings=settings, embedder=embedder, llm=llm, reranker=reranker)
    state.rebuild_from_chunks([])  # start with an empty-but-valid index
    state.load_state()
    return state
