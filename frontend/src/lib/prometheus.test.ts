import { groupSum, histogramMean, histogramQuantile, parsePrometheus, sum } from './prometheus'

const TEXT = `# HELP rag_http_requests_total HTTP requests
# TYPE rag_http_requests_total counter
rag_http_requests_total{method="POST",route="/api/v1/search",status="200"} 7.0
rag_http_requests_total{method="GET",route="/api/v1/documents",status="200"} 3.0
rag_http_requests_total{method="POST",route="/api/v1/search",status="422"} 1.0
rag_index_documents 4.0
rag_retrieval_duration_seconds_bucket{mode="bm25",le="0.01"} 6.0
rag_retrieval_duration_seconds_bucket{mode="bm25",le="0.1"} 10.0
rag_retrieval_duration_seconds_bucket{mode="bm25",le="+Inf"} 10.0
rag_retrieval_duration_seconds_sum{mode="bm25"} 0.2
rag_retrieval_duration_seconds_count{mode="bm25"} 10.0
weird{label="a \\"quoted\\" value"} 1
`

describe('parsePrometheus', () => {
  const s = parsePrometheus(TEXT)
  it('skips comments and parses labels and values', () => {
    expect(s.find((x) => x.name === 'rag_index_documents')?.value).toBe(4)
    expect(s.filter((x) => x.name === 'rag_http_requests_total')).toHaveLength(3)
    expect(s.find((x) => x.name === 'weird')?.labels.label).toBe('a "quoted" value')
  })
  it('sums, filters and groups', () => {
    expect(sum(s, 'rag_http_requests_total')).toBe(11)
    expect(sum(s, 'rag_http_requests_total', { status: '200' })).toBe(10)
    expect(groupSum(s, 'rag_http_requests_total', 'route')[0]).toEqual({ key: '/api/v1/search', value: 8 })
  })
  it('computes histogram mean and interpolated quantiles', () => {
    expect(histogramMean(s, 'rag_retrieval_duration_seconds', { mode: 'bm25' })).toBeCloseTo(0.02)
    // Prometheus-style linear interpolation: p50 = rank 5 of 10, inside the first bucket (0, 0.01] holding 6 -> 0.01 * 5/6
    expect(histogramQuantile(s, 'rag_retrieval_duration_seconds', 0.5, { mode: 'bm25' })).toBeCloseTo(0.01 * (5 / 6), 6)
    // p95 = rank 9.5, inside (0.01, 0.1] holding 4 more -> 0.01 + 0.09 * 3.5/4
    expect(histogramQuantile(s, 'rag_retrieval_duration_seconds', 0.95, { mode: 'bm25' })).toBeCloseTo(0.01 + 0.09 * (3.5 / 4), 6)
    expect(histogramQuantile(s, 'rag_retrieval_duration_seconds', 0.5, { mode: 'dense' })).toBeUndefined()
  })
  it('ignores malformed lines', () => {
    expect(parsePrometheus('garbage line\nname notanumber')).toEqual([])
  })
})
