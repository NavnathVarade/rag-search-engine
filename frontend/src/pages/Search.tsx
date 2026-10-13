import { useMutation } from '@tanstack/react-query'
import { ChevronDown, Copy, Search as SearchIcon } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { Highlight } from '@/components/Highlight'
import { ScoreStrip } from '@/components/ScoreStrip'
import { Button, buttonVariants } from '@/components/ui/button'
import { Field, Input, Select } from '@/components/ui/form'
import { Badge, Card, EmptyState, ErrorState, PageHeader, Skeleton } from '@/components/ui/primitives'
import { api, type RetrievalMode, type SearchResultItem } from '@/lib/api'
import { useInfo } from '@/hooks/queries'
import { fmtMs, MODE_LABELS, queryTerms, sourceName } from '@/lib/utils'

const MODES: RetrievalMode[] = ['hybrid', 'bm25', 'tfidf', 'dense', 'hybrid_weighted']

function Result({ r, terms, scale, minDense }: { r: SearchResultItem; terms: string[]; scale: number; minDense: number }) {
  const [open, setOpen] = useState(false)
  // Found only by vector search, and below the server's own evidence threshold: likely noise.
  const weak = !r.bm25 && r.dense != null && minDense > 0 && (r.dense.score ?? 0) < minDense
  const long = r.content.length > 420
  const source = sourceName(String(r.metadata.source ?? ''))
  return (
    <li>
      <Card className="p-5">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="flex h-6 min-w-6 items-center justify-center rounded-md bg-accent-wash px-1.5 text-xs font-semibold tabular-nums text-accent">{r.rank}</span>
          <span className="truncate text-sm font-medium">{String(r.metadata.title || source)}</span>
          <span className="truncate text-xs text-faint">{source}</span>
          {weak && <Badge tone="warn" title={`Vector similarity ${r.dense?.score?.toFixed(3)} is below the server's evidence threshold of ${minDense}, and no keyword matched.`}>Weak match</Badge>}
          <Button variant="ghost" size="icon" className="ml-auto h-7 w-7" aria-label="Copy passage" onClick={() => { navigator.clipboard?.writeText(r.content); toast.success('Passage copied') }}><Copy className="h-3.5 w-3.5" /></Button>
        </div>
        <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">
          <Highlight text={open || !long ? r.content : r.content.slice(0, 420) + '…'} terms={terms} />
        </p>
        {long && <button onClick={() => setOpen((o) => !o)} className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"><ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />{open ? 'Show less' : 'Show full passage'}</button>}
        <div className="mt-4 border-t pt-3"><ScoreStrip r={r} scale={scale} /></div>
        {r.explanation.length > 0 && (
          <details className="mt-3 text-sm"><summary className="cursor-pointer text-soft hover:text-ink">Why this result</summary>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-soft">{r.explanation.map((e, i) => <li key={i}>{e}</li>)}</ul></details>
        )}
        <p className="mt-2 font-mono text-[11px] text-faint">{r.chunk_id}</p>
      </Card>
    </li>
  )
}

export default function Search() {
  const info = useInfo()
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('hybrid')
  const [topK, setTopK] = useState(10)
  const run = useMutation({ mutationFn: () => api.search(query.trim(), mode, topK) })
  const data = run.data
  const terms = useMemo(() => queryTerms(data?.query ?? ''), [data?.query])
  const t = data?.timings

  return (
    <>
      <PageHeader title="Search" description="Run a query and see exactly how each retriever ranked every passage. Switch modes to compare keyword, vector and fused results." />
      <form onSubmit={(e) => { e.preventDefault(); if (query.trim()) run.mutate() }} role="search" className="space-y-4">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" aria-hidden />
            <Input aria-label="Search query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. how does a container restart after failure" className="h-11 pl-9 text-[15px]" maxLength={2000} autoFocus />
          </div>
          <Button type="submit" size="lg" loading={run.isPending} disabled={!query.trim()}>Search</Button>
        </div>
        <div className="grid gap-4 sm:grid-cols-[1fr_9rem]">
          <Field label="Retrieval mode">{(id) => <Select id={id} value={mode} onChange={(e) => setMode(e.target.value as RetrievalMode)}>{MODES.map((m) => <option key={m} value={m}>{MODE_LABELS[m]}</option>)}</Select>}</Field>
          <Field label="Results">{(id) => <Select id={id} value={topK} onChange={(e) => setTopK(Number(e.target.value))}>{[5, 10, 20, 50].map((n) => <option key={n} value={n}>{n}</option>)}</Select>}</Field>
        </div>
      </form>

      <div className="mt-6" aria-live="polite">
        {run.isPending && <ul className="space-y-3">{[0, 1, 2].map((n) => <li key={n}><Skeleton className="h-36" /></li>)}</ul>}
        {run.isError && <ErrorState error={run.error} onRetry={() => run.mutate()} />}
        {!run.isPending && !run.isError && !data && <Card><EmptyState title="Ask the index anything" body="Results show the matched passage, which retriever found it, and the scores behind its position." /></Card>}
        {data && !run.isPending && (
          <>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-sm text-soft">
              <span><b className="text-ink">{data.results.length}</b> {data.results.length === 1 ? 'result' : 'results'} in {t ? fmtMs(t.total_ms) : '–'}</span>
              <Badge tone="accent">{MODE_LABELS[data.retrieval_mode] ?? data.retrieval_mode}</Badge>
              {t && <span className="font-mono text-xs text-faint">bm25 {fmtMs(t.bm25_ms)} · vector {fmtMs(t.dense_ms)} · fusion {fmtMs(t.fusion_ms)}</span>}
              {data.degraded.map((d) => <Badge key={d} tone="warn">degraded: {d}</Badge>)}
            </div>
            {data.results.length === 0
              ? <Card><EmptyState title="No passages matched" body="Try different words, or switch to another retrieval mode. If the index is empty, add some content first." action={<Link to="/add" className={buttonVariants({ variant: 'outline' })}>Add content</Link>} /></Card>
              : <ul className="space-y-3">{data.results.map((r) => <Result key={r.chunk_id} r={r} terms={terms} scale={Math.max(10, data.results.length * 2)} minDense={info.data?.config.min_dense_similarity ?? 0} />)}</ul>}
          </>
        )}
      </div>
    </>
  )
}
