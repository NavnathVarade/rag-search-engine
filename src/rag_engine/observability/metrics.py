"""Prometheus metrics (spec #62 observability).

One CollectorRegistry per app (never the global default) so multiple apps/tests in one
process cannot collide on registration. Label values are low-cardinality by construction:
HTTP routes use the matched route TEMPLATE (`/api/v1/documents/{document_id}`), never the raw
path; unmatched paths collapse into "unmatched"; tenant/user/query text is never a label.

Exposed series:
  rag_http_requests_total{method,route,status}        counter
  rag_http_request_duration_seconds{method,route}     histogram
  rag_retrieval_duration_seconds{mode}                histogram  (end-to-end retrieval, from engine timings)
  rag_retrieval_stage_duration_seconds{stage}         histogram  (dense|bm25|fusion)
  rag_chat_requests_total{sufficient_evidence}        counter
  rag_llm_tokens_total{kind}                          counter    (prompt|completion)
  rag_ingestion_documents_total{outcome}              counter    (ingested|duplicate)
  rag_ingestion_chunks_total                          counter
  rag_auth_failures_total                             counter
  rag_rate_limited_total{scope}                       counter    (identity|auth_failure)
  rag_index_documents / rag_index_chunks / rag_index_vocabulary_terms   gauges (read at scrape time)
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_RETRIEVAL_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class _EngineCollector(Collector):
    """Index-size gauges computed from the live engine at scrape time (always current,
    nothing to keep in sync on rebuild/delete)."""

    def __init__(self, get_engine: Callable[[], Any]) -> None:
        self._get_engine = get_engine

    def collect(self) -> Iterable[Metric]:
        engine = self._get_engine()
        docs = GaugeMetricFamily("rag_index_documents", "Documents in the active in-memory index")
        chunks = GaugeMetricFamily("rag_index_chunks", "Chunks in the active in-memory index")
        vocab = GaugeMetricFamily("rag_index_vocabulary_terms", "Distinct terms in the BM25 inverted index")
        if engine is not None:
            docs.add_metric([], float(engine.document_count))
            chunks.add_metric([], float(len(engine.chunk_store)))
            vocab.add_metric([], float(engine.inverted_index.vocabulary_size))
        return [docs, chunks, vocab]


class Metrics:
    def __init__(self, get_engine: Callable[[], Any] = lambda: None) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.http_requests = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=r)
        self.http_duration = Histogram("rag_http_request_duration_seconds", "HTTP request latency", ["method", "route"], buckets=_HTTP_BUCKETS, registry=r)
        self.retrieval_duration = Histogram("rag_retrieval_duration_seconds", "Retrieval latency", ["mode"], buckets=_RETRIEVAL_BUCKETS, registry=r)
        self.retrieval_stage = Histogram("rag_retrieval_stage_duration_seconds", "Retrieval stage latency", ["stage"], buckets=_RETRIEVAL_BUCKETS, registry=r)
        self.chat_requests = Counter("rag_chat_requests_total", "Chat (RAG) requests", ["sufficient_evidence"], registry=r)
        self.llm_tokens = Counter("rag_llm_tokens_total", "LLM tokens", ["kind"], registry=r)
        self.ingest_documents = Counter("rag_ingestion_documents_total", "Documents ingested", ["outcome"], registry=r)
        self.ingest_chunks = Counter("rag_ingestion_chunks_total", "Chunks produced by ingestion", registry=r)
        self.auth_failures = Counter("rag_auth_failures_total", "Failed authentications", registry=r)
        self.rate_limited = Counter("rag_rate_limited_total", "Requests rejected by the rate limiter", ["scope"], registry=r)
        r.register(_EngineCollector(get_engine))

    def observe_retrieval(self, mode: str, timings: Any) -> None:
        self.retrieval_duration.labels(mode=mode).observe(timings.total_ms / 1000)
        for stage, ms in (("dense", timings.dense_ms), ("bm25", timings.bm25_ms), ("fusion", timings.fusion_ms)):
            if ms:
                self.retrieval_stage.labels(stage=stage).observe(ms / 1000)

    def render(self) -> bytes:
        return generate_latest(self.registry)
