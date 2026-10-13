import { CheckCircle2, RefreshCw, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/form'
import { Badge, Card, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton } from '@/components/ui/primitives'
import { useHealth, useMetrics, useReady } from '@/hooks/queries'
import { ApiError } from '@/lib/api'

const CHECK_LABELS: Record<string, string> = {
  index_built: 'Search index built', embedder_available: 'Embedder available', retriever_available: 'Retriever available', database: 'Database reachable',
}

function Check({ ok, label, detail }: { ok: boolean; label: string; detail?: string }) {
  return (
    <li className="flex items-center gap-3 px-5 py-3">
      {ok ? <CheckCircle2 className="h-4 w-4 text-ok" aria-hidden /> : <XCircle className="h-4 w-4 text-bad" aria-hidden />}
      <span className="flex-1 text-sm">{label}</span>
      {detail && <span className="text-xs text-faint">{detail}</span>}
      <Badge tone={ok ? 'ok' : 'bad'}>{ok ? 'Passing' : 'Failing'}</Badge>
    </li>
  )
}

export default function Health() {
  const health = useHealth()
  const ready = useReady()
  const metrics = useMetrics()
  const [filter, setFilter] = useState('')
  const rows = useMemo(() => (metrics.data ?? []).filter((s) => s.name.startsWith('rag_') && !s.name.endsWith('_bucket') && `${s.name}${JSON.stringify(s.labels)}`.toLowerCase().includes(filter.toLowerCase())), [metrics.data, filter])
  const merr = metrics.error instanceof ApiError ? metrics.error : null
  const refreshAll = () => { health.refetch(); ready.refetch(); metrics.refetch() }

  return (
    <>
      <PageHeader title="Health" description="Liveness, readiness and the raw Prometheus metrics, straight from the API." actions={<Button variant="outline" onClick={refreshAll}><RefreshCw className="h-4 w-4" />Refresh</Button>} />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Liveness" description="GET /health: the process is up." />
          {health.isLoading ? <div className="p-5"><Skeleton className="h-6" /></div> : health.isError ? <div className="p-4"><ErrorState error={health.error} onRetry={() => health.refetch()} /></div>
            : <ul><Check ok={health.data?.status === 'ok'} label="API process responding" /></ul>}
        </Card>
        <Card>
          <CardHeader title="Readiness" description="GET /ready: dependencies are usable." />
          {ready.isLoading ? <div className="space-y-2 p-5"><Skeleton className="h-6" /><Skeleton className="h-6" /></div> : ready.isError ? <div className="p-4"><ErrorState error={ready.error} onRetry={() => ready.refetch()} /></div>
            : <ul className="divide-y">{Object.entries(ready.data?.checks ?? {}).map(([k, v]) => <Check key={k} ok={v} label={CHECK_LABELS[k] ?? k} />)}</ul>}
          {ready.data?.checks && !('database' in ready.data.checks) && <p className="border-t px-5 py-3 text-xs text-faint">No database check is shown because this server runs without STORE_BACKEND=postgres.</p>}
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader title="Prometheus metrics" description="GET /metrics, parsed. Histogram buckets are hidden; see Analytics for latency." action={<Input aria-label="Filter metrics" placeholder="Filter" value={filter} onChange={(e) => setFilter(e.target.value)} className="w-40" />} />
        {metrics.isLoading ? <div className="space-y-2 p-5"><Skeleton className="h-6" /><Skeleton className="h-6" /><Skeleton className="h-6" /></div>
          : merr?.kind === 'unauthorized' ? <EmptyState title="Metrics need a token" body="This server sets METRICS_TOKEN. Enter it under Settings, in the Metrics token field." />
          : merr?.kind === 'not_found' ? <EmptyState title="Metrics are switched off" body="The server was started with METRICS_ENABLED=false." />
          : metrics.isError ? <div className="p-4"><ErrorState error={metrics.error} onRetry={() => metrics.refetch()} /></div>
          : rows.length === 0 ? <EmptyState title="No matching series" body="Clear the filter to see all series." />
          : <div className="max-h-[420px] overflow-auto"><table className="w-full text-sm"><thead className="sticky top-0 bg-surface"><tr className="border-b text-left text-soft"><th className="px-5 py-2.5 font-medium">Series</th><th className="px-5 py-2.5 text-right font-medium">Value</th></tr></thead>
              <tbody className="divide-y font-mono text-xs">{rows.map((s, i) => <tr key={i}><td className="px-5 py-2 break-all">{s.name}{Object.keys(s.labels).length > 0 && <span className="text-faint">{`{${Object.entries(s.labels).map(([k, v]) => `${k}="${v}"`).join(', ')}}`}</span>}</td><td className="px-5 py-2 text-right tabular-nums">{Number.isInteger(s.value) ? s.value : s.value.toFixed(4)}</td></tr>)}</tbody></table></div>}
      </Card>
    </>
  )
}
