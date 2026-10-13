import { ChevronLeft, ChevronRight, Eye, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { Button, buttonVariants } from '@/components/ui/button'
import { Confirm, Drawer } from '@/components/ui/dialogs'
import { Input } from '@/components/ui/form'
import { Badge, Card, EmptyState, ErrorState, PageHeader, Skeleton } from '@/components/ui/primitives'
import { useDeleteDocument, useDocument, useDocuments } from '@/hooks/queries'
import { describeError, type DocumentRow } from '@/lib/api'
import { fmtBytes, plural, sourceName } from '@/lib/utils'

const PAGE = 10

function Inspector({ doc, onClose }: { doc: DocumentRow | null; onClose: () => void }) {
  const q = useDocument(doc?.document_id ?? null)
  return (
    <Drawer open={!!doc} onOpenChange={(o) => !o && onClose()} title={doc?.title || sourceName(doc?.source) || 'Document'} description={doc?.source ?? undefined}>
      {q.isLoading && <div className="space-y-3"><Skeleton className="h-24" /><Skeleton className="h-24" /></div>}
      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
      {q.data && (
        <div className="space-y-4">
          <p className="font-mono text-xs text-faint">{q.data.document_id}</p>
          <p className="text-sm text-soft">{plural(q.data.chunk_count, 'chunk')}. This is the text the search index holds.</p>
          <ol className="space-y-3">{q.data.chunks.map((c, i) => (
            <li key={c.chunk_id} className="rounded-lg border bg-raised/50 p-3">
              <p className="mb-1.5 font-mono text-[11px] text-faint">chunk {i + 1} · {c.chunk_id}</p>
              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{c.content}</p>
            </li>))}</ol>
        </div>
      )}
    </Drawer>
  )
}

export default function Documents() {
  const docs = useDocuments()
  const del = useDeleteDocument()
  const [filter, setFilter] = useState('')
  const [page, setPage] = useState(0)
  const [inspect, setInspect] = useState<DocumentRow | null>(null)
  const [toDelete, setToDelete] = useState<DocumentRow | null>(null)

  const rows = useMemo(() => {
    const f = filter.trim().toLowerCase()
    return (docs.data?.documents ?? []).filter((d) => !f || `${d.title ?? ''} ${d.source ?? ''} ${d.document_id}`.toLowerCase().includes(f))
  }, [docs.data, filter])
  const pages = Math.max(1, Math.ceil(rows.length / PAGE))
  const safePage = Math.min(page, pages - 1)
  const view = rows.slice(safePage * PAGE, safePage * PAGE + PAGE)

  const confirmDelete = () => {
    if (!toDelete) return
    del.mutate(toDelete.document_id, {
      onSuccess: () => { toast.success(`Deleted ${sourceName(toDelete.source)}`); setToDelete(null) },
      onError: (e) => { toast.error(describeError(e)); setToDelete(null) },
    })
  }

  return (
    <>
      <PageHeader title="Documents" description="Everything you can read in the index. Deleting a document removes it from search immediately." actions={<Link to="/add" className={buttonVariants()}>Add content</Link>} />
      <Card>
        <div className="flex items-center gap-3 border-b p-4">
          <div className="relative max-w-sm flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" aria-hidden />
            <Input aria-label="Filter documents" value={filter} onChange={(e) => { setFilter(e.target.value); setPage(0) }} placeholder="Filter by name or source" className="pl-9" />
          </div>
          {docs.data && <span className="text-sm text-soft">{rows.length} of {docs.data.total}</span>}
        </div>

        {docs.isLoading ? <div className="space-y-2 p-4">{[0, 1, 2, 3].map((n) => <Skeleton key={n} className="h-11" />)}</div>
          : docs.isError ? <div className="p-4"><ErrorState error={docs.error} onRetry={() => docs.refetch()} /></div>
          : docs.data?.total === 0 ? <EmptyState title="No documents yet" body="Add a file, web page or some text and it appears here." action={<Link to="/add" className={buttonVariants()}>Add content</Link>} />
          : rows.length === 0 ? <EmptyState title="No documents match" body={`Nothing matches “${filter}”.`} action={<Button variant="outline" onClick={() => setFilter('')}>Clear filter</Button>} />
          : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Indexed documents</caption>
                <thead><tr className="border-b text-left text-soft"><th scope="col" className="px-4 py-2.5 font-medium">Document</th><th scope="col" className="hidden px-4 py-2.5 font-medium sm:table-cell">Chunks</th><th scope="col" className="hidden px-4 py-2.5 font-medium sm:table-cell">Size</th><th scope="col" className="hidden px-4 py-2.5 font-medium md:table-cell">Access</th><th scope="col" className="px-4 py-2.5 text-right font-medium">Actions</th></tr></thead>
                <tbody className="divide-y">{view.map((d) => (
                  <tr key={d.document_id} className="hover:bg-raised/60">
                    <td className="max-w-[11rem] px-4 py-3 sm:max-w-[18rem]"><p className="truncate font-medium">{d.title || sourceName(d.source)}</p><p className="truncate text-xs text-faint">{d.source}</p><p className="text-xs text-faint sm:hidden">{plural(d.chunk_count, 'chunk')}</p></td>
                    <td className="hidden px-4 py-3 tabular-nums sm:table-cell">{d.chunk_count}</td>
                    <td className="hidden px-4 py-3 text-soft sm:table-cell">{fmtBytes(d.characters)}</td>
                    <td className="hidden px-4 py-3 md:table-cell"><Badge tone={d.visibility === 'public' ? 'neutral' : 'accent'}>{d.visibility}</Badge></td>
                    <td className="px-4 py-3"><div className="flex justify-end gap-1">
                      <Button variant="ghost" size="sm" onClick={() => setInspect(d)} aria-label={`Inspect ${sourceName(d.source)}`}><Eye className="h-4 w-4" /><span className="hidden sm:inline">Inspect</span></Button>
                      {d.can_delete && <Button variant="ghost" size="icon" className="text-bad hover:text-bad" onClick={() => setToDelete(d)} aria-label={`Delete ${sourceName(d.source)}`}><Trash2 className="h-4 w-4" /></Button>}
                    </div></td>
                  </tr>))}</tbody>
              </table>
            </div>
          )}

        {rows.length > PAGE && (
          <div className="flex items-center justify-between border-t px-4 py-3 text-sm">
            <span className="text-soft">Page {safePage + 1} of {pages}</span>
            <div className="flex gap-1">
              <Button variant="outline" size="sm" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}><ChevronLeft className="h-4 w-4" /> Previous</Button>
              <Button variant="outline" size="sm" disabled={safePage >= pages - 1} onClick={() => setPage(safePage + 1)}>Next <ChevronRight className="h-4 w-4" /></Button>
            </div>
          </div>
        )}
      </Card>

      <Inspector doc={inspect} onClose={() => setInspect(null)} />
      <Confirm open={!!toDelete} onOpenChange={(o) => !o && setToDelete(null)} title="Delete this document?" confirmLabel="Delete document" busy={del.isPending} onConfirm={confirmDelete}
        body={<>“{sourceName(toDelete?.source)}” and its {plural(toDelete?.chunk_count ?? 0, 'chunk')} will no longer appear in search or answers. You can add it again later.</>} />
    </>
  )
}
