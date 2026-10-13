import { useMemo } from 'react'

/** Wraps query terms in <mark>. Terms are escaped before use in a RegExp. */
export function Highlight({ text, terms }: { text: string; terms: string[] }) {
  const parts = useMemo(() => {
    if (!terms.length) return [text]
    const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
    return text.split(re)
  }, [text, terms])
  const set = new Set(terms)
  return (
    <>
      {parts.map((p, i) => (set.has(p.toLowerCase()) ? <mark key={i} className="rounded-sm bg-mark px-0.5 text-ink">{p}</mark> : <span key={i}>{p}</span>))}
    </>
  )
}
