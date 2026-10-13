# Observability

## Structured logging (`logging.py`)
`structlog`-based; JSON output in production (`LOG_JSON=true`), readable
console output in development. Sensitive keys (`api_key`, `authorization`,
`password`, `token`, `secret`, `llm_api_key`) are redacted by a processor
before any record is emitted, regardless of where in the call stack they
were logged from.

## Request correlation (`api/middleware.py::RequestContextMiddleware`)
Every request gets a `request_id` (from `X-Request-Id` header, or a fresh
UUID), attached to `request.state` and echoed back in the response
header, and logged with method/path/status/duration on completion.

## Metrics (`GET /metrics`, Prometheus text format)
Served when `METRICS_ENABLED=true` (default). If `METRICS_TOKEN` is set the scraper must send
`Authorization: Bearer <token>`. Implemented with `prometheus_client`
(`observability/metrics.py`); series: `rag_http_requests_total{method,route,status}`,
`rag_http_request_duration_seconds`, `rag_retrieval_duration_seconds{mode}`,
`rag_retrieval_stage_duration_seconds{stage}`, `rag_chat_requests_total{sufficient_evidence}`,
`rag_llm_tokens_total{kind}`, `rag_ingestion_documents_total{outcome}`,
`rag_ingestion_chunks_total`, `rag_auth_failures_total`, `rag_rate_limited_total{scope}`, and
index gauges `rag_index_documents`, `rag_index_chunks`, `rag_index_vocabulary_terms`.
Verified live: counters increment per request and `/metrics` returns 401 without the token.
Not done: a Grafana dashboard, alert rules, and per-tenant series (deliberately avoided to keep
label cardinality bounded).

## Tracing
`config.py::otel_endpoint` is the configuration hook for OpenTelemetry
export; this reference build does not instrument spans itself. The
natural instrumentation points are already timed and separated in the
codebase (`HybridSearchTimings.dense_ms/bm25_ms/fusion_ms`, reranking
duration in `api/routes/chat.py`, generation duration in
`GenerationResult`), so wrapping each in an OTel span is a mechanical
addition rather than a redesign.

## What "explainability" means in this project vs "observability"
These are deliberately separate concerns. Observability (this doc) is
about operating the *system* -- is it slow, is it erroring, what's the
request volume. Explainability (docs/hybrid-search.md, spec #105) is
about a single *query* -- why did we retrieve these specific chunks. The
latter is fully implemented (every `RetrievalResult` carries a
programmatically-generated explanation); the former's metrics/tracing
wiring is the more mechanical, lower-priority remaining piece.
