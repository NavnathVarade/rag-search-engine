import { AlertTriangle, Inbox, RefreshCw } from 'lucide-react'
import type { HTMLAttributes, ReactNode } from 'react'
import { describeError } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Button } from './button'

export const Card = ({ className, ...p }: HTMLAttributes<HTMLDivElement>) => <div className={cn('rounded-xl border bg-surface', className)} {...p} />

export function CardHeader({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b px-5 py-4">
      <div className="min-w-0">
        <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
        {description && <p className="mt-0.5 text-sm text-soft">{description}</p>}
      </div>
      {action}
    </div>
  )
}

type Tone = 'neutral' | 'ok' | 'warn' | 'bad' | 'accent'
const tones: Record<Tone, string> = {
  neutral: 'bg-raised text-soft border',
  ok: 'bg-ok/10 text-ok border border-ok/25',
  warn: 'bg-warn/10 text-warn border border-warn/30',
  bad: 'bg-bad/10 text-bad border border-bad/25',
  accent: 'bg-accent-wash text-accent border border-accent/25',
}
export const Badge = ({ tone = 'neutral', className, ...p }: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) => (
  <span className={cn('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium', tones[tone], className)} {...p} />
)

export const Skeleton = ({ className }: { className?: string }) => <div aria-hidden className={cn('animate-pulse rounded-md bg-raised', className)} />

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-soft">{description}</p>}
      </div>
      {actions}
    </div>
  )
}

export function EmptyState({ title, body, action, icon }: { title: string; body: string; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center px-6 py-14 text-center">
      <div className="mb-3 rounded-full bg-raised p-3 text-faint">{icon ?? <Inbox className="h-5 w-5" />}</div>
      <p className="font-medium">{title}</p>
      <p className="mt-1 max-w-sm text-sm text-soft">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-bad/30 bg-bad/5 p-4">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bad" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-bad">Request failed</p>
        <p className="mt-0.5 text-sm text-soft">{describeError(error)}</p>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </Button>
      )}
    </div>
  )
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: Tone }) {
  return (
    <Card className="p-5">
      <p className="text-sm text-soft">{label}</p>
      <p className={cn('mt-1 text-3xl font-semibold tabular-nums tracking-tight', tone === 'bad' && 'text-bad', tone === 'ok' && 'text-ok')}>{value}</p>
      {hint && <p className="mt-1 text-xs text-faint">{hint}</p>}
    </Card>
  )
}
