import { authHeaders, getConnection } from './auth'

/* ---------- types: mirror src/rag_engine/api/schemas and routes exactly ---------- */
export type RetrievalMode = 'hybrid' | 'bm25' | 'tfidf' | 'dense' | 'hybrid_weighted'
export interface RankerDetail { rank: number | null; score: number | null }
export interface SearchResultItem {
  chunk_id: string; document_id: string; content: string; rank: number; score: number; method: string
  dense: RankerDetail | null; bm25: RankerDetail | null; rrf_score: number | null; reranker_score: number | null
  explanation: string[]; metadata: Record<string, unknown>
}
export interface SearchTimings { embedding_ms: number; dense_ms: number; bm25_ms: number; fusion_ms: number; reranking_ms: number; total_ms: number }
export interface SearchResponse { query: string; retrieval_mode: string; results: SearchResultItem[]; timings: SearchTimings; degraded: string[] }
export interface Citation { marker: number; chunk_id: string; document_id: string | null; source_label: string }
export interface ChatResponse {
  question: string; answer: string; citations: Citation[]; retrieved_chunks: SearchResultItem[]; had_sufficient_evidence: boolean
  injection_flags: string[]; model: string; prompt_tokens: number; completion_tokens: number; timings: SearchTimings
}
export interface DocumentRow {
  document_id: string; source: string | null; title: string | null; visibility: string; owner: string | null
  chunk_count: number; characters: number; can_delete: boolean
}
export interface DocumentDetail {
  document_id: string; chunk_count: number; chunk_ids: string[]; source: string | null; title: string | null
  chunks: { chunk_id: string; content: string }[]
}
export interface IngestResult {
  document_id: string; was_duplicate: boolean; chunk_count: number; stage_timings_ms: Record<string, number>
  source_uri: string | null; title: string | null
}
export interface Info {
  version: string
  caller: { user_id: string | null; tenant_id: string | null; groups: string[]; is_admin: boolean }
  index: { built: boolean; documents: number; chunks: number; total_documents: number; total_chunks: number; scope: string }
  config: {
    environment: string; auth_enabled: boolean; store_backend: string; state_file: boolean
    embedding_provider: string; embedding_model: string; embedding_dimension: number
    bm25_stemmer: string; reranker_enabled: boolean; min_dense_similarity: number
    chunking_strategy: string; chunk_max_tokens: number
    llm_configured: boolean; llm_provider: string; llm_model: string
    max_document_size_mb: number; url_ingest_allow_private: boolean; metrics_enabled: boolean; metrics_requires_token: boolean
  }
}
export interface BenchmarkConfigResult {
  'ndcg@10': number; 'recall@10': number; 'recall@100': number; 'mrr@10': number; latency_p50_ms?: number; latency_p95_ms?: number
}
export interface BenchmarkReport {
  file: string; dataset: string; split: string; num_docs: number; num_queries: number; embedder: string
  hnsw_recall_at_10_vs_bruteforce: number | null; results: Record<string, BenchmarkConfigResult>
  weight_tuning?: { split: string | null; chosen_dense_weight: number | null; 'ndcg@10_by_dense_weight': Record<string, number> }
}
export interface Readiness { status: string; checks: Record<string, boolean> }

/* ---------- errors ---------- */
export type ErrorKind = 'network' | 'unauthorized' | 'forbidden' | 'not_found' | 'rate_limited' | 'unavailable' | 'client' | 'server'
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public kind: ErrorKind, public retryAfter?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

function kindFor(status: number): ErrorKind {
  if (status === 401) return 'unauthorized'
  if (status === 403) return 'forbidden'
  if (status === 404) return 'not_found'
  if (status === 429) return 'rate_limited'
  if (status === 503) return 'unavailable'
  return status >= 500 ? 'server' : 'client'
}

/** Plain-language message for an error, telling the person what to do next. */
export function describeError(e: unknown): string {
  if (!(e instanceof ApiError)) return e instanceof Error ? e.message : 'Something went wrong.'
  switch (e.kind) {
    case 'network': return 'Cannot reach the API. Check that the server is running and that the API address in Settings is correct.'
    case 'unauthorized': return 'The server rejected your credentials. Open Settings and enter a valid API key or token.'
    case 'forbidden': return 'You do not have permission to do this. It needs an administrator key.'
    case 'rate_limited': return `Too many requests. Try again${e.retryAfter ? ` in ${e.retryAfter} seconds` : ' in a moment'}.`
    case 'unavailable':
      return e.code === 'persistence_failed' ? 'The database write failed, so nothing was changed. Check that Postgres is running, then try again.' : `The service is not ready: ${e.message}`
    default: return e.message
  }
}

/* ---------- transport ---------- */
async function send(path: string, init: RequestInit = {}, extraHeaders: Record<string, string> = {}): Promise<Response> {
  const conn = getConnection()
  const headers = { ...authHeaders(conn), ...extraHeaders, ...((init.headers as Record<string, string>) ?? {}) }
  let res: Response
  try {
    res = await fetch(`${conn.baseUrl}${path}`, { ...init, headers })
  } catch {
    throw new ApiError(0, 'network', 'network error', 'network')
  }
  if (res.ok) return res
  let code = 'http_error'
  let message = res.statusText || `HTTP ${res.status}`
  try {
    const body = await res.json()
    if (typeof body?.error === 'string') { code = body.error; message = typeof body.message === 'string' ? body.message : message }
    else if (Array.isArray(body?.detail)) { code = 'validation_error'; message = body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join('; ') || message }
  } catch { /* non-JSON error body */ }
  const retry = Number(res.headers.get('retry-after'))
  throw new ApiError(res.status, code, message, kindFor(res.status), Number.isFinite(retry) && retry > 0 ? retry : undefined)
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await send(path, init)).json() as Promise<T>
}
const post = <T>(path: string, body: unknown) =>
  json<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

/* ---------- endpoints ---------- */
export const api = {
  info: () => json<Info>('/api/v1/info'),
  health: () => json<{ status: string }>('/health'),
  ready: async (): Promise<Readiness> => {
    // /ready answers 503 with a JSON body when something is down; that body is the data we want.
    const conn = getConnection()
    try {
      const res = await fetch(`${conn.baseUrl}/ready`)
      return (await res.json()) as Readiness
    } catch {
      throw new ApiError(0, 'network', 'network error', 'network')
    }
  },
  metricsText: async (): Promise<string> => {
    const token = getConnection().metricsToken
    const res = await send('/metrics', {}, token ? { Authorization: `Bearer ${token}` } : {})
    return res.text()
  },
  search: (query: string, retrieval_mode: RetrievalMode, top_k: number) => post<SearchResponse>('/api/v1/search', { query, retrieval_mode, top_k }),
  chat: (question: string, retrieval_mode: RetrievalMode, top_k: number) => post<ChatResponse>('/api/v1/chat', { question, retrieval_mode, top_k }),
  listDocuments: () => json<{ documents: DocumentRow[]; total: number }>('/api/v1/documents'),
  getDocument: (id: string) => json<DocumentDetail>(`/api/v1/documents/${encodeURIComponent(id)}`),
  deleteDocument: (id: string) => json<{ document_id: string; deleted: boolean }>(`/api/v1/documents/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  upload: (file: File | Blob, name: string, visibility: string) => {
    const form = new FormData()
    form.append('file', file, name)
    form.append('visibility', visibility)
    return json<IngestResult>('/api/v1/documents/upload', { method: 'POST', body: form })
  },
  ingestUrl: (url: string, visibility: string) => post<IngestResult>('/api/v1/documents/ingest-url', { url, visibility }),
  rebuildIndex: () => json<{ rebuilt: boolean; document_count: number; chunk_count: number }>('/api/v1/index/rebuild', { method: 'POST' }),
  benchmarks: () => json<{ reports: BenchmarkReport[] }>('/api/v1/benchmarks'),
}
