import { KeyRound, WifiOff } from 'lucide-react'
import type { ReactNode } from 'react'
import { useInfo } from '@/hooks/queries'
import { ApiError } from '@/lib/api'
import { ConnectionForm } from './ConnectionForm'
import { Card } from './ui/primitives'
import { Skeleton } from './ui/primitives'
import { Button } from './ui/button'

/** Renders the app only once /api/v1/info succeeds. Distinguishes "cannot reach the server" from "credentials rejected". */
export function AuthGate({ children }: { children: ReactNode }) {
  const info = useInfo()
  if (info.isLoading) return <div className="mx-auto max-w-md space-y-3 p-10"><Skeleton className="h-8 w-48" /><Skeleton className="h-40" /></div>
  if (info.isError) {
    const err = info.error
    const unauthorized = err instanceof ApiError && err.kind === 'unauthorized'
    const network = err instanceof ApiError && err.kind === 'network'
    const Icon = unauthorized ? KeyRound : WifiOff
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <Card className="w-full max-w-lg p-6">
          <div className="mb-4 flex items-start gap-3">
            <div className="rounded-full bg-raised p-2.5 text-soft"><Icon className="h-5 w-5" aria-hidden /></div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">{unauthorized ? 'Sign in to the API' : network ? 'Cannot reach the API' : 'The API returned an error'}</h1>
              <p className="mt-1 text-sm text-soft">
                {unauthorized
                  ? 'This server requires authentication. Enter an API key (create one with `rag auth create-key`) or a bearer token.'
                  : network
                    ? 'Start the server (for example `uvicorn rag_engine.api.app:app`) or correct the address below.'
                    : err instanceof Error ? err.message : 'Unknown error.'}
              </p>
            </div>
          </div>
          <ConnectionForm compact />
          <div className="mt-4 border-t pt-4"><Button variant="ghost" size="sm" onClick={() => info.refetch()}>Retry without changes</Button></div>
        </Card>
      </div>
    )
  }
  return <>{children}</>
}
