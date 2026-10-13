/** Minimal parser for the Prometheus text exposition format (what GET /metrics returns). */
export interface Sample { name: string; labels: Record<string, string>; value: number }

export function parsePrometheus(text: string): Sample[] {
  const out: Sample[] = []
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const m = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)/.exec(line)
    if (!m) continue
    const labels: Record<string, string> = {}
    if (m[3]) for (const l of m[3].matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g)) labels[l[1]] = l[2].replace(/\\(["\\n])/g, (_, c) => (c === 'n' ? '\n' : c))
    const value = m[4] === '+Inf' ? Infinity : m[4] === '-Inf' ? -Infinity : Number(m[4])
    if (!Number.isNaN(value)) out.push({ name: m[1], labels, value })
  }
  return out
}

export const sum = (s: Sample[], name: string, where: Record<string, string> = {}) =>
  s.filter((x) => x.name === name && Object.entries(where).every(([k, v]) => x.labels[k] === v)).reduce((a, x) => a + x.value, 0)

export const first = (s: Sample[], name: string) => s.find((x) => x.name === name)?.value

export function groupSum(s: Sample[], name: string, label: string): { key: string; value: number }[] {
  const m = new Map<string, number>()
  for (const x of s) if (x.name === name) m.set(x.labels[label] ?? '', (m.get(x.labels[label] ?? '') ?? 0) + x.value)
  return [...m].map(([key, value]) => ({ key, value })).sort((a, b) => b.value - a.value)
}

/** Mean latency in seconds from a histogram's _sum and _count (undefined when no observations). */
export function histogramMean(s: Sample[], base: string, where: Record<string, string> = {}): number | undefined {
  const n = sum(s, `${base}_count`, where)
  return n > 0 ? sum(s, `${base}_sum`, where) / n : undefined
}

/** Approximate quantile from cumulative buckets (linear interpolation, as Prometheus does). */
export function histogramQuantile(s: Sample[], base: string, q: number, where: Record<string, string> = {}): number | undefined {
  const buckets = new Map<number, number>()
  for (const x of s) {
    if (x.name !== `${base}_bucket` || !Object.entries(where).every(([k, v]) => x.labels[k] === v)) continue
    const le = x.labels.le === '+Inf' ? Infinity : Number(x.labels.le)
    buckets.set(le, (buckets.get(le) ?? 0) + x.value)
  }
  const sorted = [...buckets].sort((a, b) => a[0] - b[0])
  const total = sorted.at(-1)?.[1] ?? 0
  if (total === 0) return undefined
  const rank = q * total
  let prevLe = 0
  let prevCount = 0
  for (const [le, count] of sorted) {
    if (count >= rank) {
      if (!Number.isFinite(le)) return prevLe
      return count === prevCount ? le : prevLe + ((le - prevLe) * (rank - prevCount)) / (count - prevCount)
    }
    prevLe = le
    prevCount = count
  }
  return undefined
}
