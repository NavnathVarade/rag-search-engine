import { Database, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Confirm } from '@/components/ui/dialogs'
import { Badge, Card, CardHeader, ErrorState, PageHeader, Skeleton, Stat } from '@/components/ui/primitives'
import { useInfo, useRebuildIndex } from '@/hooks/queries'
import { describeError } from '@/lib/api'
import { fmtNumber } from '@/lib/utils'

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return <div className="flex items-baseline justify-between gap-4 px-5 py-2.5 text-sm"><dt className="text-soft">{k}</dt><dd className="text-right font-medium">{v}</dd></div>
}

export default function IndexPage() {
  const info = useInfo()
  const rebuild = useRebuildIndex()
  const [open, setOpen] = useState(false)
  if (info.isError) return <ErrorState error={info.error} onRetry={() => info.refetch()} />
  const i = info.data
  const c = i?.config
  const postgres = c?.store_backend === 'postgres'

  const go = () =>
    rebuild.mutate(undefined, {
      onSuccess: (r) => { toast.success(`Index rebuilt: ${r.document_count} documents, ${r.chunk_count} chunks`); setOpen(false) },
      onError: (e) => { toast.error(describeError(e)); setOpen(false) },
    })

  return (
    <>
      <PageHeader title="Index" description="The in-memory search indexes (keyword, vector, TF-IDF) and how they were built." actions={
        <Button variant="outline" disabled={!i?.caller.is_admin} onClick={() => setOpen(true)} title={i && !i.caller.is_admin ? 'Needs an administrator key' : undefined}><RefreshCw className="h-4 w-4" />Rebuild index</Button>} />
      {i && !i.caller.is_admin && <p className="-mt-3 mb-5 text-sm text-soft">Rebuilding needs an administrator credential. You are signed in as a regular user.</p>}

      <div className="grid gap-4 sm:grid-cols-3">
        {!i ? [0, 1, 2].map((n) => <Skeleton key={n} className="h-[104px]" />) : (
          <>
            <Stat label="Documents" value={fmtNumber(i.index.total_documents)} hint="In the active index (all tenants)" />
            <Stat label="Chunks" value={fmtNumber(i.index.total_chunks)} hint="Passages searched" />
            <Stat label="State" value={i.index.built ? 'Built' : 'Empty'} tone={i.index.built ? 'ok' : 'bad'} hint={postgres ? 'Rebuilt from Postgres at startup' : 'Rebuilt from memory'} />
          </>
        )}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Retrieval configuration" />
          {!c ? <div className="space-y-2 p-5"><Skeleton className="h-5" /><Skeleton className="h-5" /><Skeleton className="h-5" /></div> : (
            <dl className="divide-y">
              <Row k="Keyword ranking" v="BM25 (Okapi)" />
              <Row k="Stemming" v={c.bm25_stemmer} />
              <Row k="Vector embedder" v={<>{c.embedding_provider} <span className="font-normal text-faint">· {c.embedding_dimension}d</span></>} />
              <Row k="Fusion" v="Reciprocal Rank Fusion" />
              <Row k="Reranker" v={c.reranker_enabled ? 'Heuristic reranker on' : 'Off'} />
              <Row k="Chunking" v={`${c.chunking_strategy}, up to ${c.chunk_max_tokens} tokens`} />
              <Row k="Evidence gate" v={c.min_dense_similarity > 0 ? `vector similarity ≥ ${c.min_dense_similarity}` : 'Off'} />
            </dl>
          )}
        </Card>
        <Card>
          <CardHeader title="Storage and consistency" />
          {!c ? <div className="p-5"><Skeleton className="h-24" /></div> : (
            <div className="space-y-4 p-5 text-sm">
              <div className="flex items-center gap-2"><Database className="h-4 w-4 text-accent" aria-hidden /><Badge tone={postgres ? 'ok' : 'warn'}>{postgres ? 'Postgres is the source of truth' : 'No database'}</Badge></div>
              <p className="text-soft">{postgres ? 'Every add and delete is written to the database first. If that write fails nothing changes and the API answers 503. On restart the indexes are rebuilt from the database.' : c.state_file ? 'Documents are kept in memory and saved to a JSON snapshot file. Set STORE_BACKEND=postgres for durable storage.' : 'Documents live only in memory and disappear on restart. Set STORE_BACKEND=postgres, or STATE_FILE for a simple snapshot.'}</p>
              <p className="rounded-lg border border-warn/30 bg-warn/5 p-3 text-soft"><b className="font-medium text-ink">One instance only.</b> Each API process builds its own index in memory and instances do not synchronise, so a second replica would serve stale results.</p>
              <p className="text-soft">Rebuild reloads {postgres ? 'every chunk from Postgres' : 'the in-memory records'} and recreates all three indexes. Use it if you changed the database directly.</p>
            </div>
          )}
        </Card>
      </div>

      <Confirm open={open} onOpenChange={setOpen} title="Rebuild the search index?" confirmLabel="Rebuild index" busy={rebuild.isPending} onConfirm={go}
        body="Searches keep working, but results may be briefly slower while the index is recreated. Nothing is deleted." />
    </>
  )
}
