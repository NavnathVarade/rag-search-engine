import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs))

export const fmtNumber = (n: number) => new Intl.NumberFormat('en').format(n)
export const fmtMs = (ms: number) => (ms >= 100 ? `${Math.round(ms)} ms` : ms >= 10 ? `${ms.toFixed(1)} ms` : `${ms.toFixed(2)} ms`)
export const fmtBytes = (chars: number) => (chars >= 1_000_000 ? `${(chars / 1_000_000).toFixed(1)}M chars` : chars >= 1000 ? `${(chars / 1000).toFixed(1)}k chars` : `${chars} chars`)

export const MODE_LABELS: Record<string, string> = {
  hybrid: 'Hybrid (BM25 + vector, RRF)',
  bm25: 'BM25 keywords',
  tfidf: 'TF-IDF keywords',
  dense: 'Vector similarity',
  hybrid_weighted: 'Hybrid (weighted)',
}

const STOP = new Set('a an and are as at be by for from how in is it of on or that the this to was what when where which who why with do does can i you we'.split(' '))

/** Query terms worth highlighting in result text (lower-cased, stop words and 1-letter words removed). */
export function queryTerms(query: string): string[] {
  return [...new Set(query.toLowerCase().match(/[a-z0-9_]+/g) ?? [])].filter((t) => t.length > 1 && !STOP.has(t))
}

export function sourceName(source: string | null | undefined): string {
  if (!source) return 'inline'
  try {
    const u = new URL(source)
    return (u.hostname + u.pathname).replace(/\/$/, '')
  } catch {
    return source.split('/').pop() || source
  }
}

export const plural = (n: number, one: string, many = `${one}s`) => `${fmtNumber(n)} ${n === 1 ? one : many}`
