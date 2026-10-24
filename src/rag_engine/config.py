"""Centralized, environment-driven configuration.

All tunables referenced throughout the spec (BM25_K1, RRF_K, DENSE_TOP_K,
etc.) are declared here as a single Pydantic Settings object so that:
  * nothing is hardcoded in algorithm modules,
  * config is validated at startup (fail fast on bad values),
  * every experiment/run can serialize its exact configuration for
    reproducibility (see evaluation/experiment.py).
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    test = "test"
    production = "production"


class RetrievalMode(str, Enum):
    dense = "dense"
    bm25 = "bm25"
    tfidf = "tfidf"
    hybrid = "hybrid"
    hybrid_weighted = "hybrid_weighted"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- environment ---
    environment: Environment = Environment.development
    log_level: str = "INFO"
    log_json: bool = False

    # --- database ---
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    database_pool_size: int = 10
    database_max_overflow: int = 5
    database_pool_timeout_s: int = 30
    database_pool_recycle_s: int = 1800

    # --- embeddings ---
    embedding_provider: str = "hashing"  # "hashing" (offline, deterministic) | "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_cache_size: int = 10_000

    # --- sparse / BM25 ---
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_stemmer: str = "porter"  # "none", "light" or "porter" (measured +2.6 nDCG@10 pts on SciFact)

    # --- ANN ---
    ann_backend: str = "brute_force"  # "brute_force" | "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # --- hybrid retrieval ---
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    default_retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    # --- reranking ---
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    rerank_candidate_k: int = 50

    # --- context selection ---
    max_context_tokens: int = 6000
    context_similarity_dedup_threshold: float = 0.92
    # Evidence gate for chat: a chunk counts as evidence only if it has a lexical (BM25/TF-IDF) match or
    # a dense cosine similarity >= this value. Embedder-specific: ~0.2 suits the offline hashing
    # embedder; real sentence embedders usually want 0.3+. Set 0 to disable the dense requirement.
    min_dense_similarity: float = 0.2
    max_chunks_per_document: int = 3

    # --- chunking ---
    chunking_strategy: str = "recursive"
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # --- generation ---
    llm_provider: str = "openai_compatible"  # "openai" (alias: "openai_compatible") | "local" (Ollama) | "none"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3

    # --- resource limits ---
    max_document_size_mb: int = 25
    max_pdf_pages: int = 500
    max_repo_size_mb: int = 500
    max_files_per_repo: int = 5000
    max_query_length: int = 2000
    max_concurrent_ingestion_jobs: int = 4
    request_timeout_s: float = 60.0
    # Operator override for POST /documents/ingest-url: allow private/loopback targets (SSRF protection off). Keep false in production.
    url_ingest_allow_private: bool = False
    # Where the API finds committed benchmark JSON files and the built frontend (served at / when present).
    benchmarks_dir: str = "benchmarks/results"
    frontend_dir: str = "frontend/dist"

    # --- security ---
    auth_enabled: bool = False
    # JWT bearer tokens: `jwt_key` is the HMAC secret (HS*) or the PEM public key (RS*/ES*).
    jwt_key: str = ""
    jwt_algorithms: tuple[str, ...] = ("HS256",)
    jwt_issuer: str = ""  # if set, the token's `iss` must match
    jwt_audience: str = ""  # if set, the token's `aud` must match
    # API keys: JSON object {sha256_hex_of_key: {"user_id": ..., "tenant_id": ..., "groups": [...], "admin": false}}.
    # Only hashes are stored; generate with `rag auth create-key`.
    api_keys_json: str = ""
    metrics_token: str = ""  # if set, /metrics requires `Authorization: Bearer <token>`
    allowed_url_schemes: tuple[str, ...] = ("https", "http")
    blocked_ip_ranges: tuple[str, ...] = (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "169.254.0.0/16", "::1/128", "fc00::/7",
    )
    rate_limit_anonymous_per_min: int = 30
    rate_limit_authenticated_per_min: int = 120

    # --- observability ---
    otel_endpoint: str = ""
    metrics_enabled: bool = True

    # --- storage ---
    # "none": memory (+ optional STATE_FILE snapshot). "postgres": documents/chunks are stored in
    # Postgres (DATABASE_URL; run `alembic upgrade head` first) and the index is rebuilt from it at startup.
    store_backend: str = "none"

    # --- paths ---
    # JSON snapshot of the API's index. Empty = memory only (lost on restart). Docker compose sets it.
    state_file: str = ""
    data_dir: Path = Path("data")

    @field_validator("store_backend")
    @classmethod
    def _check_store_backend(cls, v: str) -> str:
        if v not in ("none", "postgres"):
            raise ValueError("store_backend must be 'none' or 'postgres'")
        return v

    @field_validator("bm25_k1")
    @classmethod
    def _k1_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("bm25_k1 must be > 0")
        return v

    @field_validator("bm25_b")
    @classmethod
    def _b_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("bm25_b must be in [0, 1]")
        return v

    @field_validator("jwt_algorithms")
    @classmethod
    def _no_none_alg(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v or any(a.lower() == "none" for a in v):
            raise ValueError("jwt_algorithms must be non-empty and must not include 'none'")
        return v

    @field_validator("rrf_k")
    @classmethod
    def _rrf_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rrf_k must be > 0")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
