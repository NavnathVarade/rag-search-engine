import { useMutation } from '@tanstack/react-query'
import { AlertTriangle, ChevronDown, SendHorizonal, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Link } from 'react-router-dom'
import { Highlight } from '@/components/Highlight'
import { Button, buttonVariants } from '@/components/ui/button'
import { Select, Textarea } from '@/components/ui/form'
import { Badge, Card, EmptyState, ErrorState } from '@/components/ui/primitives'
import { useInfo } from '@/hooks/queries'
import { api, type ChatResponse, type RetrievalMode } from '@/lib/api'
import { fmtMs, MODE_LABELS, queryTerms, sourceName } from '@/lib/utils'

interface Turn { id: number; question: string; reply?: ChatResponse; error?: unknown; pending?: boolean }

const SUGGESTIONS = ['How do I restart a failed container?', 'What does BM25 use to rank documents?', 'How is authentication handled?']

/** Replaces literal [n] markers in text nodes with buttons that open the matching source. */
function withCitations(children: ReactNode, onCite: (n: number) => void): ReactNode {
  const walk = (node: ReactNode): ReactNode => {
    if (typeof node === 'string') {
      return node.split(/(\[\d+\])/g).map((part, i) => {
        const m = /^\[(\d+)\]$/.exec(part)
        return m ? <button key={i} onClick={() => onCite(Number(m[1]))} className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-accent-wash px-1 align-baseline font-sans text-[11px] font-semibold text-accent hover:bg-accent hover:text-accent-ink" aria-label={`Open source ${m[1]}`}>{m[1]}</button> : part
      })
    }
    if (Array.isArray(node)) return node.map((n, i) => <span key={i}>{walk(n)}</span>)
    return node
  }
  return walk(children)
}

function Answer({ turn }: { turn: Turn }) {
  const r = turn.reply as ChatResponse
  const [openId, setOpenId] = useState<string | null>(null)
  const terms = useMemo(() => queryTerms(r.question), [r.question])
  const byMarker = new Map(r.citations.map((c) => [c.marker, c]))
  const stub = r.model === 'null'
  const cite = (n: number) => { const c = byMarker.get(n); if (c) { setOpenId(c.chunk_id); document.getElementById(`src-${turn.id}-${c.chunk_id}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) } }
  const wrap = (Tag: 'p' | 'li') => ({ children }: { children?: ReactNode }) => <Tag>{withCitations(children, cite)}</Tag>

  return (
    <div className="space-y-4">
      {!r.had_sufficient_evidence && (
        <div role="status" className="flex gap-3 rounded-lg border border-warn/30 bg-warn/5 p-3.5">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" aria-hidden />
          <p className="text-sm"><b className="font-medium">Not enough evidence.</b> <span className="text-soft">Nothing in the index matched this question well enough to answer from, so no model was asked. Add relevant documents or rephrase.</span></p>
        </div>
      )}
      {stub && r.had_sufficient_evidence && (
        <div role="status" className="flex gap-3 rounded-lg border border-warn/30 bg-warn/5 p-3.5">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" aria-hidden />
          <p className="text-sm text-soft"><b className="font-medium text-ink">Offline stub answer.</b> No language model is configured, so this is not a written answer. The cited passages below are the real retrieved evidence.</p>
        </div>
      )}
      <div className="prose-answer"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: wrap('p'), li: wrap('li') }}>{r.answer}</ReactMarkdown></div>
      {r.injection_flags.length > 0 && <Badge tone="warn">Suspicious instructions in sources were ignored: {r.injection_flags.join(', ')}</Badge>}

      {r.citations.length > 0 && (
        <div>
          <p className="mb-2 text-sm font-medium">Sources</p>
          <ul className="space-y-2">
            {r.citations.map((c) => {
              const chunk = r.retrieved_chunks.find((x) => x.chunk_id === c.chunk_id)
              const open = openId === c.chunk_id
              return (
                <li key={c.chunk_id} id={`src-${turn.id}-${c.chunk_id}`} className="rounded-lg border bg-raised/50">
                  <button className="flex w-full items-center gap-3 px-3 py-2.5 text-left" aria-expanded={open} onClick={() => setOpenId(open ? null : c.chunk_id)}>
                    <span className="flex h-5 min-w-5 items-center justify-center rounded bg-accent-wash px-1 text-[11px] font-semibold text-accent">{c.marker}</span>
                    <span className="min-w-0 flex-1 truncate text-sm">{sourceName(c.source_label)}</span>
                    <ChevronDown className={`h-4 w-4 shrink-0 text-faint transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
                  </button>
                  {open && <div className="border-t px-3 py-3 text-sm leading-relaxed">{chunk ? <p className="whitespace-pre-wrap break-words"><Highlight text={chunk.content} terms={terms} /></p> : <p className="text-soft">The passage text was not returned.</p>}</div>}
                </li>
              )
            })}
          </ul>
        </div>
      )}
      <p className="font-mono text-[11px] text-faint">{stub ? 'no model' : r.model} · {r.retrieved_chunks.length} {r.retrieved_chunks.length === 1 ? 'passage' : 'passages'} retrieved · {fmtMs(r.timings.total_ms)}{r.prompt_tokens ? ` · ${r.prompt_tokens}+${r.completion_tokens} tokens` : ''}</p>
    </div>
  )
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('hybrid')
  const info = useInfo()
  const end = useRef<HTMLDivElement>(null)
  const ask = useMutation({
    mutationFn: ({ q }: { id: number; q: string }) => api.chat(q, mode, 50),
    onSuccess: (reply, { id }) => setTurns((t) => t.map((x) => (x.id === id ? { ...x, reply, pending: false } : x))),
    onError: (error, { id }) => setTurns((t) => t.map((x) => (x.id === id ? { ...x, error, pending: false } : x))),
  })
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [turns])

  const submit = (q: string) => {
    const text = q.trim()
    if (!text || ask.isPending) return
    const id = Date.now()
    setTurns((t) => [...t, { id, question: text, pending: true }])
    setQuestion('')
    ask.mutate({ id, q: text })
  }
  const empty = info.data && info.data.index.chunks === 0

  return (
    <div className="mx-auto flex max-w-3xl flex-col">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Ask</h1>
          <p className="mt-1 text-sm text-soft">Answers are built only from your indexed documents and cite their sources. Each question is answered on its own; earlier turns are not used as context.</p>
        </div>
        {turns.length > 0 && <Button variant="ghost" size="sm" onClick={() => setTurns([])}><Trash2 className="h-3.5 w-3.5" /> Clear conversation</Button>}
      </div>

      {empty && <Card className="mb-4"><EmptyState title="The index is empty" body="There is nothing to answer from yet." action={<Link to="/add" className={buttonVariants()}>Add content</Link>} /></Card>}
      {turns.length === 0 && !empty && (
        <Card className="p-6">
          <p className="font-medium">Try a question</p>
          <div className="mt-3 flex flex-wrap gap-2">{SUGGESTIONS.map((s) => <button key={s} onClick={() => submit(s)} className="rounded-full border bg-surface px-3 py-1.5 text-sm text-soft hover:bg-raised hover:text-ink">{s}</button>)}</div>
        </Card>
      )}

      <ol className="space-y-6" aria-live="polite">
        {turns.map((t) => (
          <li key={t.id} className="space-y-3">
            <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[15px] text-accent-ink">{t.question}</div>
            <Card className="p-5">
              {t.pending && <div className="space-y-2"><div className="h-3 w-3/4 animate-pulse rounded bg-raised" /><div className="h-3 w-1/2 animate-pulse rounded bg-raised" /><span className="sr-only">Searching and answering…</span></div>}
              {t.error != null && <ErrorState error={t.error} onRetry={() => { setTurns((x) => x.filter((y) => y.id !== t.id)); submit(t.question) }} />}
              {t.reply && <Answer turn={t} />}
            </Card>
          </li>
        ))}
      </ol>
      <div ref={end} />

      <form onSubmit={(e) => { e.preventDefault(); submit(question) }} className="sticky bottom-0 mt-6 space-y-2 bg-bg pb-2 pt-3">
        <div className="flex items-end gap-2">
          <Textarea aria-label="Your question" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(question) } }} placeholder="Ask about your documentation…  (Enter to send, Shift+Enter for a new line)" className="min-h-[48px] resize-none" rows={1} maxLength={2000} />
          <Button type="submit" size="lg" loading={ask.isPending} disabled={!question.trim()} aria-label="Send question"><SendHorizonal className="h-4 w-4" /></Button>
        </div>
        <div className="flex items-center gap-2 text-xs text-faint"><label htmlFor="chat-mode">Retrieval</label><Select id="chat-mode" className="h-7 w-auto py-0 text-xs" value={mode} onChange={(e) => setMode(e.target.value as RetrievalMode)}>{(['hybrid', 'bm25', 'tfidf', 'dense', 'hybrid_weighted'] as const).map((m) => <option key={m} value={m}>{MODE_LABELS[m]}</option>)}</Select></div>
      </form>
    </div>
  )
}
