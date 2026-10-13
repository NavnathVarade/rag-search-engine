import { BarChartCard, type Series } from '@/components/BarChartCard'
import { Badge, Card, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton, Stat } from '@/components/ui/primitives'
import { useBenchmarks, useMetrics } from '@/hooks/queries'
import { ApiError, type BenchmarkReport } from '@/lib/api'
import { groupSum, histogramMean, histogramQuantile, sum, type Sample } from '@/lib/prometheus'
import { fmtNumber } from '@/lib/utils'

/** Route templates from the metrics, in plain words. */
const routeName = (k: string) => (k === '/{path:path}' ? 'web app files' : k.replace('/api/v1', '') || '/')

const S1 = 'var(--series-1)'
const S2 = 'var(--series-2)'
const SHORT_MODE: Record<string, string> = { hybrid: 'Hybrid (RRF)', hybrid_weighted: 'Hybrid (weighted)', bm25: 'BM25 keywords', tfidf: 'TF-IDF keywords', dense: 'Vector similarity' }
const MODES = ['hybrid', 'bm25', 'tfidf', 'dense', 'hybrid_weighted']
const CONFIG_NAMES: Record<string, string> = {
  bm25: 'BM25', dense_bruteforce: 'Vector (exact)', hybrid_bruteforce: 'Hybrid (exact)', hybrid_weighted_tuned: 'Hybrid weighted', dense_hnsw: 'Vector (HNSW)', hybrid_hnsw: 'Hybrid (HNSW)',
}

function LiveMetrics({ samples }: { samples: Sample[] }) {
  const routes = groupSum(samples, 'rag_http_requests_total', 'route').filter((r) => !['/metrics', '/ready', '/health', 'unmatched'].includes(r.key)).slice(0, 8)
  const latency = MODES.map((m) => {
    const where = { mode: m }
    const p50 = histogramQuantile(samples, 'rag_retrieval_duration_seconds', 0.5, where)
    const p95 = histogramQuantile(samples, 'rag_retrieval_duration_seconds', 0.95, where)
    return { name: SHORT_MODE[m] ?? m, n: sum(samples, 'rag_retrieval_duration_seconds_count', where), p50: p50 == null ? null : p50 * 1000, p95: p95 == null ? null : p95 * 1000 }
  }).filter((x) => x.n > 0)
  const stages = ['embedding', 'dense', 'bm25', 'fusion', 'reranking'].map((st) => ({ name: st, mean: histogramMean(samples, 'rag_retrieval_stage_duration_seconds', { stage: st }) }))
  const added = sum(samples, 'rag_ingestion_documents_total', { outcome: 'ingested' })
  const dup = sum(samples, 'rag_ingestion_documents_total', { outcome: 'duplicate' })
  const chatOk = sum(samples, 'rag_chat_requests_total', { sufficient_evidence: 'true' })
  const chatNo = sum(samples, 'rag_chat_requests_total', { sufficient_evidence: 'false' })
  const requestsTotal = sum(samples, 'rag_http_requests_total')

  return (
    <div className="space-y-6">
      <p className="text-sm text-soft">These counters start at zero each time the API process starts, and cover only this process.</p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="HTTP requests" value={fmtNumber(requestsTotal)} />
        <Stat label="Documents added" value={fmtNumber(added)} hint={`${fmtNumber(dup)} duplicates skipped`} />
        <Stat label="Questions answered" value={fmtNumber(chatOk)} hint={`${fmtNumber(chatNo)} without enough evidence`} />
        <Stat label="Auth failures / rate limited" value={`${fmtNumber(sum(samples, 'rag_auth_failures_total'))} / ${fmtNumber(sum(samples, 'rag_rate_limited_total'))}`} tone={sum(samples, 'rag_auth_failures_total') > 0 ? 'bad' : undefined} />
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        {routes.length > 0 ? <BarChartCard title="Requests by endpoint" data={routes.map((r) => ({ name: routeName(r.key), requests: r.value }))} horizontal series={[{ key: 'requests', label: 'Requests', color: S1 }]} format={(n) => fmtNumber(n)} />
          : <Card><CardHeader title="Requests by endpoint" /><EmptyState title="No API traffic yet" body="Run a search or add a document and refresh." /></Card>}
        {latency.length > 0 ? <BarChartCard title="Search latency by mode" description="Estimated from histogram buckets since the process started." data={latency} series={[{ key: 'p50', label: 'p50 (ms)', color: S1 }, { key: 'p95', label: 'p95 (ms)', color: S2 }]} format={(n) => `${n.toFixed(n < 10 ? 1 : 0)} ms`} horizontal />
          : <Card><CardHeader title="Search latency by mode" /><EmptyState title="No searches yet" body="Latency appears once searches have run." /></Card>}
      </div>
      {stages.some((s) => s.mean != null) && (
        <BarChartCard title="Average time per search stage" data={stages.filter((s) => s.mean != null).map((s) => ({ name: s.name, mean: (s.mean as number) * 1000 }))} series={[{ key: 'mean', label: 'Mean (ms)', color: S1 }]} format={(n) => `${n.toFixed(n < 10 ? 2 : 1)} ms`} horizontal />
      )}
    </div>
  )
}

function Benchmarks({ reports }: { reports: BenchmarkReport[] }) {
  const names = [...new Set(reports.flatMap((r) => Object.keys(r.results)))]
  const series: Series[] = reports.slice(0, 2).map((r, i) => ({ key: r.dataset, label: r.dataset, color: i === 0 ? S1 : S2 }))
  const ndcg = names.map((n) => ({ name: CONFIG_NAMES[n] ?? n, ...Object.fromEntries(reports.map((r) => [r.dataset, r.results[n]?.['ndcg@10'] ?? null])) }))
  const recall = names.map((n) => ({ name: CONFIG_NAMES[n] ?? n, ...Object.fromEntries(reports.map((r) => [r.dataset, r.results[n]?.['recall@100'] ?? null])) }))
  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <BarChartCard title="Ranking quality (nDCG@10)" description="Higher is better. Measured on public BEIR test queries." data={ndcg} series={series} format={(n) => n.toFixed(3)} domain={[0, 1]} horizontal />
        <BarChartCard title="Coverage (recall@100)" description="Share of relevant documents found in the top 100." data={recall} series={series} format={(n) => n.toFixed(3)} domain={[0, 1]} horizontal />
      </div>
      <Card>
        <CardHeader title="Benchmark runs" description="Exactly as written by scripts/run_beir.py." />
        <div className="divide-y">{reports.map((r) => (
          <div key={r.file} className="space-y-3 p-5">
            <div className="flex flex-wrap items-center gap-2"><b className="font-semibold">{r.dataset}</b><Badge>{fmtNumber(r.num_docs)} documents</Badge><Badge>{fmtNumber(r.num_queries)} queries</Badge><Badge tone="warn">embedder: {r.embedder}</Badge>{r.hnsw_recall_at_10_vs_bruteforce != null && <Badge tone="accent">HNSW recall@10 vs exact: {r.hnsw_recall_at_10_vs_bruteforce}</Badge>}</div>
            <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr className="border-b text-left text-soft"><th className="py-2 pr-4 font-medium">Method</th><th className="py-2 pr-4 text-right font-medium">nDCG@10</th><th className="py-2 pr-4 text-right font-medium">Recall@10</th><th className="py-2 pr-4 text-right font-medium">Recall@100</th><th className="py-2 pr-4 text-right font-medium">MRR@10</th><th className="py-2 text-right font-medium">p50 / p95 ms</th></tr></thead>
              <tbody className="divide-y font-mono tabular-nums">{Object.entries(r.results).map(([k, v]) => <tr key={k}><td className="py-2 pr-4 font-sans">{CONFIG_NAMES[k] ?? k}</td><td className="py-2 pr-4 text-right">{v['ndcg@10']}</td><td className="py-2 pr-4 text-right">{v['recall@10']}</td><td className="py-2 pr-4 text-right">{v['recall@100']}</td><td className="py-2 pr-4 text-right">{v['mrr@10']}</td><td className="py-2 text-right">{v.latency_p50_ms ?? '–'} / {v.latency_p95_ms ?? '–'}</td></tr>)}</tbody></table></div>
            {r.weight_tuning?.chosen_dense_weight != null && <p className="text-sm text-soft">Dense weight tuned on the “{r.weight_tuning.split}” split: best value was {r.weight_tuning.chosen_dense_weight}. A value of 0 means vector search added nothing over BM25 with this embedder.</p>}
          </div>))}</div>
      </Card>
      <p className="text-sm text-soft">These are offline benchmark results, not live traffic. They used the hashing embedder, which does not capture meaning, so they say nothing about how a real embedding model would perform.</p>
    </div>
  )
}

export default function Analytics() {
  const metrics = useMetrics()
  const bench = useBenchmarks()
  const metricsErr = metrics.error instanceof ApiError ? metrics.error : null

  return (
    <>
      <PageHeader title="Analytics" description="Live counters from the API's Prometheus metrics, and the recorded retrieval benchmarks. Nothing here is estimated by the UI." />
      <section aria-labelledby="live" className="mb-10">
        <h2 id="live" className="mb-3 text-lg font-semibold tracking-tight">Live activity</h2>
        {metrics.isLoading ? <div className="grid gap-4 sm:grid-cols-4">{[0, 1, 2, 3].map((n) => <Skeleton key={n} className="h-[104px]" />)}</div>
          : metricsErr?.kind === 'unauthorized' ? <Card><EmptyState title="Metrics need a token" body="This server sets METRICS_TOKEN. Enter it under Settings to see live counters." /></Card>
          : metricsErr?.kind === 'not_found' ? <Card><EmptyState title="Metrics are switched off" body="The server was started with METRICS_ENABLED=false." /></Card>
          : metrics.isError ? <ErrorState error={metrics.error} onRetry={() => metrics.refetch()} />
          : metrics.data && <LiveMetrics samples={metrics.data} />}
      </section>
      <section aria-labelledby="bench">
        <h2 id="bench" className="mb-3 text-lg font-semibold tracking-tight">Retrieval benchmarks</h2>
        {bench.isLoading ? <Skeleton className="h-72" />
          : bench.isError ? <ErrorState error={bench.error} onRetry={() => bench.refetch()} />
          : bench.data && bench.data.reports.length === 0 ? <Card><EmptyState title="No benchmark results found" body="Run scripts/run_beir.py with --out benchmarks/results/<name>.json and they appear here." /></Card>
          : bench.data && <Benchmarks reports={bench.data.reports} />}
      </section>
    </>
  )
}
