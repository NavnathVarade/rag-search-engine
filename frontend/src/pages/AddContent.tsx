import * as Tabs from '@radix-ui/react-tabs'
import { CheckCircle2, CopyCheck, FileUp, Globe, Type, XCircle } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Field, Input, Select, Textarea } from '@/components/ui/form'
import { Badge, Card, CardHeader, PageHeader } from '@/components/ui/primitives'
import { useInfo, useInvalidateIndex } from '@/hooks/queries'
import { api, describeError, type IngestResult } from '@/lib/api'
import { cn, fmtMs, plural } from '@/lib/utils'

interface Outcome { id: number; label: string; status: 'added' | 'duplicate' | 'failed'; result?: IngestResult; error?: string }
const ACCEPT = '.pdf,.html,.htm,.md,.txt,.py,.js,.jsx,.ts,.tsx,.go,.java,.rs,.yaml,.yml,.json,.sql,.sh,.rb,.c,.cpp,.h,.hpp'

export default function AddContent() {
  const info = useInfo()
  const refresh = useInvalidateIndex()
  const [visibility, setVisibility] = useState('public')
  const [outcomes, setOutcomes] = useState<Outcome[]>([])
  const [busy, setBusy] = useState(false)
  const [drag, setDrag] = useState(false)
  const [url, setUrl] = useState('')
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const seq = useRef(0)
  const maxMb = info.data?.config.max_document_size_mb ?? 25

  const record = (o: Omit<Outcome, 'id'>) => setOutcomes((x) => [{ ...o, id: ++seq.current }, ...x])
  const run = async (label: string, job: () => Promise<IngestResult>) => {
    try {
      const result = await job()
      record({ label, status: result.was_duplicate ? 'duplicate' : 'added', result })
      if (result.was_duplicate) toast.info(`${label} is already indexed`)
      else toast.success(`Added ${label}: ${plural(result.chunk_count, 'chunk')}`)
    } catch (e) {
      record({ label, status: 'failed', error: describeError(e) })
      toast.error(`${label}: ${describeError(e)}`)
    }
  }
  const finish = async () => { await refresh(); setBusy(false) }

  const uploadFiles = async (files: FileList | File[]) => {
    setBusy(true)
    for (const f of Array.from(files)) {
      if (f.size > maxMb * 1024 * 1024) { record({ label: f.name, status: 'failed', error: `Larger than the ${maxMb} MB limit.` }); continue }
      await run(f.name, () => api.upload(f, f.name, visibility))
    }
    await finish()
  }

  const submitUrl = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true)
    await run(url.trim(), () => api.ingestUrl(url.trim(), visibility))
    setUrl(''); await finish()
  }
  const submitText = async (e: React.FormEvent) => {
    e.preventDefault(); setBusy(true)
    const name = `${(title.trim() || 'pasted-note').replace(/[^\w.\- ]+/g, '').replace(/\s+/g, '-').slice(0, 80)}.md`
    await run(name, () => api.upload(new Blob([text], { type: 'text/markdown' }), name, visibility))
    setText(''); setTitle(''); await finish()
  }

  const tab = 'inline-flex items-center gap-2 border-b-2 border-transparent px-4 py-2.5 text-sm font-medium text-soft hover:text-ink data-[state=active]:border-accent data-[state=active]:text-ink'

  return (
    <>
      <PageHeader title="Add content" description="Anything you add is parsed, split into passages and searchable right away. Content that is already indexed is detected and skipped." />
      <Card>
        <Tabs.Root defaultValue="files">
          <Tabs.List className="flex overflow-x-auto border-b px-2" aria-label="Source type">
            <Tabs.Trigger value="files" className={tab}><FileUp className="h-4 w-4" />Files</Tabs.Trigger>
            <Tabs.Trigger value="url" className={tab}><Globe className="h-4 w-4" />Web page</Tabs.Trigger>
            <Tabs.Trigger value="text" className={tab}><Type className="h-4 w-4" />Pasted text</Tabs.Trigger>
          </Tabs.List>
          <div className="space-y-5 p-5">
            <div className="max-w-xs">
              <Field label="Who can see it" hint={visibility === 'public' ? 'Everyone in your tenant.' : 'Only you (and administrators).'}>
                {(id, d) => <Select id={id} aria-describedby={d} value={visibility} onChange={(e) => setVisibility(e.target.value)}><option value="public">Public</option><option value="private">Private</option></Select>}
              </Field>
            </div>

            <Tabs.Content value="files" className="space-y-3">
              <div onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files) }}
                className={cn('flex flex-col items-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors', drag ? 'border-accent bg-accent-wash' : 'bg-raised/40')}>
                <FileUp className="mb-3 h-6 w-6 text-faint" aria-hidden />
                <p className="font-medium">Drop files here</p>
                <p className="mt-1 text-sm text-soft">PDF, HTML, Markdown, plain text and source code. Up to {maxMb} MB each.</p>
                <Button className="mt-4" variant="outline" loading={busy} onClick={() => fileInput.current?.click()}>Choose files</Button>
                <input ref={fileInput} type="file" multiple accept={ACCEPT} className="sr-only" aria-label="Choose files to upload" onChange={(e) => { if (e.target.files?.length) uploadFiles(e.target.files); e.target.value = '' }} />
              </div>
            </Tabs.Content>

            <Tabs.Content value="url">
              <form onSubmit={submitUrl} className="max-w-xl space-y-3">
                <Field label="Page address" hint={info.data?.config.url_ingest_allow_private ? 'Private network addresses are allowed on this server.' : 'Public http(s) pages and PDFs. Private and local addresses are blocked for safety.'}>
                  {(id, d) => <Input id={id} aria-describedby={d} type="url" required value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://docs.example.com/guide" />}
                </Field>
                <Button type="submit" loading={busy} disabled={!url.trim()}>Fetch and add</Button>
              </form>
            </Tabs.Content>

            <Tabs.Content value="text">
              <form onSubmit={submitText} className="max-w-2xl space-y-3">
                <Field label="Name">{(id) => <Input id={id} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. runbook-redis" maxLength={80} />}</Field>
                <Field label="Content" hint="Markdown is understood: headings become section titles.">{(id, d) => <Textarea id={id} aria-describedby={d} required value={text} onChange={(e) => setText(e.target.value)} className="min-h-[200px] font-mono" placeholder={'# Title\n\nWrite or paste documentation here.'} />}</Field>
                <Button type="submit" loading={busy} disabled={!text.trim()}>Add text</Button>
              </form>
            </Tabs.Content>
          </div>
        </Tabs.Root>
      </Card>

      <Card className="mt-6">
        <CardHeader title="Results" description="Newest first. Kept until you leave this page." action={outcomes.length > 0 ? <Button variant="ghost" size="sm" onClick={() => setOutcomes([])}>Clear</Button> : undefined} />
        {outcomes.length === 0 ? <p className="px-5 py-8 text-center text-sm text-soft">Nothing added yet in this session.</p> : (
          <ul className="divide-y" aria-live="polite">{outcomes.map((o) => (
            <li key={o.id} className="flex items-start gap-3 px-5 py-3.5">
              {o.status === 'added' && <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-ok" aria-hidden />}
              {o.status === 'duplicate' && <CopyCheck className="mt-0.5 h-4 w-4 shrink-0 text-warn" aria-hidden />}
              {o.status === 'failed' && <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-bad" aria-hidden />}
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{o.result?.title || o.label}</p>
                <p className="text-sm text-soft">
                  {o.status === 'added' && <>Added with {plural(o.result?.chunk_count ?? 0, 'chunk')}. {o.result && <span className="font-mono text-xs text-faint">parse {fmtMs(o.result.stage_timings_ms.parsing_ms ?? 0)} · chunk {fmtMs(o.result.stage_timings_ms.chunking_ms ?? 0)}</span>}</>}
                  {o.status === 'duplicate' && 'Already indexed with identical content, so nothing was added.'}
                  {o.status === 'failed' && o.error}
                </p>
              </div>
              <Badge tone={o.status === 'added' ? 'ok' : o.status === 'duplicate' ? 'warn' : 'bad'}>{o.status === 'added' ? 'Added' : o.status === 'duplicate' ? 'Duplicate' : 'Failed'}</Badge>
            </li>))}</ul>
        )}
      </Card>
    </>
  )
}
