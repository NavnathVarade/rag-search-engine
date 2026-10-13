import { AlertTriangle, CheckCircle2, MessageSquareText, Search as SearchIcon, UploadCloud } from 'lucide-react'
import { Link } from 'react-router-dom'
import { buttonVariants } from '@/components/ui/button'
import { Badge, Card, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton, Stat } from '@/components/ui/primitives'
import { useDocuments, useInfo, useReady } from '@/hooks/queries'
import { fmtNumber, plural, sourceName } from '@/lib/utils'

function Notice({ tone, title, children }: { tone: 'warn' | 'ok'; title: string; children: React.ReactNode }) {
  const Icon = tone === 'ok' ? CheckCircle2 : AlertTriangle
  return (
    <li className="flex gap-3 px-5 py-3.5">
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${tone === 'ok' ? 'text-ok' : 'text-warn'}`} aria-hidden />
      <div><p className="text-sm font-medium">{title}</p><p className="mt-0.5 text-sm text-soft">{children}</p></div>
    </li>
  )
}

export default function Dashboard() {
  const info = useInfo()
  const ready = useReady()
  const docs = useDocuments()

  if (info.isError) return <ErrorState error={info.error} onRetry={() => info.refetch()} />
  const i = info.data
  const c = i?.config
  const recent = (docs.data?.documents ?? []).slice(0, 5)

  return (
    <>
      <PageHeader title="Dashboard" description="What is indexed, whether the system is healthy, and how it is configured." actions={
        <div className="flex gap-2">
          <Link to="/add" className={buttonVariants({ variant: 'outline' })}><UploadCloud className="h-4 w-4" />Add content</Link>
          <Link to="/search" className={buttonVariants()}><SearchIcon className="h-4 w-4" />Search</Link>
        </div>
      } />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {!i ? [0, 1, 2, 3].map((n) => <Skeleton key={n} className="h-[104px]" />) : (
          <>
            <Stat label="Documents" value={fmtNumber(i.index.documents)} hint={i.index.scope} />
            <Stat label="Chunks" value={fmtNumber(i.index.chunks)} hint="Searchable passages" />
            <Stat label="Readiness" value={ready.data?.status === 'ready' ? 'Ready' : ready.isError ? 'Down' : 'Not ready'} tone={ready.data?.status === 'ready' ? 'ok' : 'bad'}
              hint={ready.data ? Object.entries(ready.data.checks).filter(([, v]) => !v).map(([k]) => k).join(', ') || 'All checks passing' : undefined} />
            <Stat label="Storage" value={c?.store_backend === 'postgres' ? 'Postgres' : 'Memory'} hint={c?.store_backend === 'postgres' ? 'Survives restarts' : c?.state_file ? 'JSON snapshot file' : 'Lost on restart'} />
          </>
        )}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader title="Documents" action={<Link to="/documents" className="text-sm text-accent hover:underline">View all</Link>} />
          {docs.isLoading ? <div className="space-y-2 p-5"><Skeleton className="h-9" /><Skeleton className="h-9" /><Skeleton className="h-9" /></div>
            : docs.isError ? <div className="p-5"><ErrorState error={docs.error} onRetry={() => docs.refetch()} /></div>
            : recent.length === 0 ? <EmptyState title="Nothing indexed yet" body="Upload a Markdown, PDF or HTML file, or point at a URL, and it becomes searchable immediately." action={<Link to="/add" className={buttonVariants()}>Add your first document</Link>} />
            : <ul className="divide-y">{recent.map((d) => (
                <li key={d.document_id} className="flex items-center justify-between gap-3 px-5 py-3">
                  <div className="min-w-0"><p className="truncate text-sm font-medium">{d.title || sourceName(d.source)}</p><p className="truncate text-xs text-faint">{sourceName(d.source)}</p></div>
                  <Badge>{plural(d.chunk_count, 'chunk')}</Badge>
                </li>))}</ul>}
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader title="Quick actions" />
          <div className="grid gap-2 p-4">
            <Link to="/chat" className="flex items-center gap-3 rounded-lg border p-3 hover:bg-raised"><MessageSquareText className="h-4 w-4 text-accent" /><span className="text-sm"><b className="font-medium">Ask a question</b><br /><span className="text-soft">Answer with citations, or an honest “not enough evidence”.</span></span></Link>
            <Link to="/search" className="flex items-center gap-3 rounded-lg border p-3 hover:bg-raised"><SearchIcon className="h-4 w-4 text-accent" /><span className="text-sm"><b className="font-medium">Inspect retrieval</b><br /><span className="text-soft">See how keyword and vector search each ranked a result.</span></span></Link>
            <Link to="/add" className="flex items-center gap-3 rounded-lg border p-3 hover:bg-raised"><UploadCloud className="h-4 w-4 text-accent" /><span className="text-sm"><b className="font-medium">Add content</b><br /><span className="text-soft">Files, web pages or pasted text. Duplicates are detected.</span></span></Link>
          </div>
        </Card>
      </div>

      {c && (
        <Card className="mt-6">
          <CardHeader title="What to know about this deployment" description="Read from the running server, including what is not production-grade yet." />
          <ul className="divide-y">
            {!c.llm_configured
              ? <Notice tone="warn" title="Answers use the offline stub">No language model is configured, so Ask returns a fixed message that points to the cited passages. Set LLM_API_KEY (or LLM_PROVIDER=local for Ollama) and restart for generated answers.</Notice>
              : <Notice tone="ok" title={`Answers generated by ${c.llm_model}`}>Provider: {c.llm_provider}. Citations are validated against the retrieved passages.</Notice>}
            {c.embedding_provider === 'hashing'
              ? <Notice tone="warn" title="Vector search uses a hashing embedder">It is deterministic and offline, but it matches shared word pieces, not meaning. Keyword (BM25) search is the stronger signal today; a real embedding model has not been validated.</Notice>
              : <Notice tone="ok" title={`Embeddings: ${c.embedding_model}`}>{c.embedding_dimension} dimensions.</Notice>}
            <Notice tone={c.auth_enabled ? 'ok' : 'warn'} title={c.auth_enabled ? 'Authentication is on' : 'Authentication is off'}>{c.auth_enabled ? 'Requests need an API key or bearer token; documents are isolated per tenant.' : 'Every caller acts as an administrator. Set AUTH_ENABLED=true before exposing this server.'}</Notice>
            <Notice tone="warn" title="Run a single API instance">The search index lives in each process's memory and is not synchronised between instances.</Notice>
          </ul>
        </Card>
      )}
    </>
  )
}
