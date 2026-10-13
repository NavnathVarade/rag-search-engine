import { render, screen } from '@testing-library/react'
import { Highlight } from './Highlight'
import { ScoreStrip } from './ScoreStrip'
import { queryTerms, sourceName } from '@/lib/utils'
import type { SearchResultItem } from '@/lib/api'

const item = (over: Partial<SearchResultItem> = {}): SearchResultItem => ({
  chunk_id: 'c1', document_id: 'd1', content: 'x', rank: 1, score: 0.0312, method: 'hybrid',
  dense: { rank: 4, score: 0.41 }, bm25: { rank: 1, score: 7.25 }, rrf_score: 0.0312, reranker_score: 0.8, explanation: [], metadata: {}, ...over,
})

describe('Highlight', () => {
  it('marks matching terms case-insensitively and leaves other text alone', () => {
    const { container } = render(<Highlight text="Kubernetes restarts a Failed container" terms={['failed', 'kubernetes']} />)
    expect([...container.querySelectorAll('mark')].map((m) => m.textContent)).toEqual(['Kubernetes', 'Failed'])
  })
  it('escapes regex metacharacters in terms', () => {
    const { container } = render(<Highlight text="use a.b (x) now" terms={['a.b', '(x)']} />)
    expect(container.querySelectorAll('mark')).toHaveLength(2)
  })
})

describe('ScoreStrip', () => {
  it('shows each ranker rank and score from the API, plus fused values', () => {
    render(<ScoreStrip r={item()} />)
    expect(screen.getByLabelText(/Keywords ranked this result number 1/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Vector ranked this result number 4/)).toBeInTheDocument()
    expect(screen.getByText('#1 · 7.250')).toBeInTheDocument()
    expect(screen.getByText('rrf 0.0312')).toBeInTheDocument()
    expect(screen.getByText('rerank 0.800')).toBeInTheDocument()
  })
  it('says so when a ranker did not return the result, and hides per-ranker bars for single-mode results', () => {
    const { rerender } = render(<ScoreStrip r={item({ dense: null })} />)
    expect(screen.getByText('not returned')).toBeInTheDocument()
    rerender(<ScoreStrip r={item({ dense: null, bm25: null, method: 'tfidf' })} />)
    expect(screen.queryByText(/Keywords/)).not.toBeInTheDocument()
  })
})

describe('utils', () => {
  it('drops stop words and short tokens from query terms', () => {
    expect(queryTerms('How does the BM25 ranking work in k8s?')).toEqual(['bm25', 'ranking', 'work', 'k8s'])
  })
  it('names sources from URLs and file paths', () => {
    expect(sourceName('https://docs.example.com/guide/')).toBe('docs.example.com/guide')
    expect(sourceName('/a/b/redis.md')).toBe('redis.md')
    expect(sourceName(null)).toBe('inline')
  })
})
