import type { SearchResultItem } from '@/lib/api'
import { cn } from '@/lib/utils'

/**
 * The signature element: shows where each ranker placed this chunk. A shorter bar means a better rank
 * (rank 1 = full bar), so a reader can see at a glance whether keyword search, vector search, or both
 * surfaced the result, and what the fused and reranked scores were. Values are the API's, not derived.
 */
function RankBar({ label, rank, score, color, scale }: { label: string; rank: number | null | undefined; score?: number | null; color: string; scale: number }) {
  const found = rank != null
  const width = found ? Math.max(6, 100 - ((rank - 1) / scale) * 94) : 0
  return (
    <div className="grid grid-cols-[4.5rem_1fr_auto] items-center gap-3 text-xs">
      <span className="text-soft">{label}</span>
      <div className="h-1.5 rounded-full bg-raised" role="img" aria-label={found ? `${label} ranked this result number ${rank}` : `${label} did not return this result`}>
        {found && <div className={cn('h-full origin-left rounded-full animate-grow', color)} style={{ width: `${width}%` }} />}
      </div>
      <span className="w-28 text-right font-mono text-[11px] tabular-nums text-soft">
        {found ? `#${rank}${score != null ? ` · ${score.toFixed(3)}` : ''}` : 'not returned'}
      </span>
    </div>
  )
}

export function ScoreStrip({ r, scale = 50 }: { r: SearchResultItem; scale?: number }) {
  const single = !r.bm25 && !r.dense
  return (
    <div className="space-y-1.5" aria-label="How each ranker scored this result">
      {!single && <RankBar label="Keywords" rank={r.bm25?.rank} score={r.bm25?.score} color="bg-sparse" scale={scale} />}
      {!single && <RankBar label="Vector" rank={r.dense?.rank} score={r.dense?.score} color="bg-dense" scale={scale} />}
      <div className="flex flex-wrap gap-x-4 gap-y-1 pt-1 font-mono text-[11px] text-soft">
        <span>final {r.score.toFixed(4)}</span>
        {r.rrf_score != null && <span>rrf {r.rrf_score.toFixed(4)}</span>}
        {r.reranker_score != null && <span>rerank {r.reranker_score.toFixed(3)}</span>}
        <span>{r.method}</span>
      </div>
    </div>
  )
}
