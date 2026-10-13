import { Eye, EyeOff } from 'lucide-react'
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { clearCredentials, saveConnection, useConnection, type CredentialKind } from '@/lib/auth'
import { Button } from './ui/button'
import { Field, Input, Select } from './ui/form'

/** Where the API lives and how to authenticate. Mirrors what the backend accepts:
 *  `x-api-key: <key>` or `Authorization: Bearer <jwt>`. With AUTH_ENABLED=false no credential is needed. */
export function ConnectionForm({ onSaved, compact }: { onSaved?: () => void; compact?: boolean }) {
  const saved = useConnection()
  const qc = useQueryClient()
  const [draft, setDraft] = useState(saved)
  const [show, setShow] = useState(false)
  const set = <K extends keyof typeof draft>(k: K, v: (typeof draft)[K]) => setDraft((d) => ({ ...d, [k]: v }))

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    saveConnection(draft)
    qc.resetQueries()
    toast.success('Connection saved')
    onSaved?.()
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <Field label="API address" hint="Leave empty to use the server that served this page (or the dev proxy). Example: http://localhost:8000">
        {(id, d) => <Input id={id} aria-describedby={d} value={draft.baseUrl} onChange={(e) => set('baseUrl', e.target.value)} placeholder="Same origin" inputMode="url" autoComplete="off" />}
      </Field>
      <div className="grid gap-4 sm:grid-cols-[11rem_1fr]">
        <Field label="Credential type">
          {(id) => (
            <Select id={id} value={draft.credentialKind} onChange={(e) => set('credentialKind', e.target.value as CredentialKind)}>
              <option value="api-key">API key</option>
              <option value="bearer">Bearer token (JWT)</option>
            </Select>
          )}
        </Field>
        <Field label={draft.credentialKind === 'api-key' ? 'API key' : 'Token'} hint="Not needed when the server runs with AUTH_ENABLED=false.">
          {(id, d) => (
            <div className="relative">
              <Input id={id} aria-describedby={d} type={show ? 'text' : 'password'} value={draft.credential} onChange={(e) => set('credential', e.target.value)} autoComplete="off" spellCheck={false} className="pr-10 font-mono" />
              <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide credential' : 'Show credential'} className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-faint hover:text-ink">
                {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          )}
        </Field>
      </div>
      {!compact && (
        <Field label="Metrics token (optional)" hint="Only if the server sets METRICS_TOKEN. Used for the /metrics endpoint on the Health and Analytics pages.">
          {(id, d) => <Input id={id} aria-describedby={d} type="password" value={draft.metricsToken} onChange={(e) => set('metricsToken', e.target.value)} autoComplete="off" className="font-mono" />}
        </Field>
      )}
      <label className="flex items-start gap-2 text-sm">
        <input type="checkbox" checked={draft.remember} onChange={(e) => set('remember', e.target.checked)} className="mt-0.5 h-4 w-4 accent-[hsl(var(--accent))]" />
        <span>
          Remember on this device
          <span className="block text-xs text-faint">Off by default: credentials are kept for this browser tab only and cleared when it closes.</span>
        </span>
      </label>
      <div className="flex flex-wrap gap-2">
        <Button type="submit">Save and connect</Button>
        {!compact && (
          <Button type="button" variant="outline" onClick={() => { clearCredentials(); setDraft((d) => ({ ...d, credential: '', metricsToken: '' })); qc.resetQueries(); toast.success('Credentials removed') }}>
            Remove credentials
          </Button>
        )}
      </div>
    </form>
  )
}
