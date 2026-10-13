import { ExternalLink } from 'lucide-react'
import { ConnectionForm } from '@/components/ConnectionForm'
import { Badge, Card, CardHeader, PageHeader } from '@/components/ui/primitives'
import { useInfo } from '@/hooks/queries'
import { getConnection } from '@/lib/auth'

export default function Settings() {
  const info = useInfo()
  const i = info.data
  const base = getConnection().baseUrl
  return (
    <>
      <PageHeader title="Settings" description="Where the API lives and how this browser signs in to it." />
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader title="Connection and credentials" description="Credentials are sent only as request headers to the address above." />
          <div className="p-5"><ConnectionForm /></div>
        </Card>
        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader title="Signed in as" />
            {i ? (
              <dl className="space-y-2.5 p-5 text-sm">
                <div className="flex justify-between gap-3"><dt className="text-soft">User</dt><dd className="font-medium">{i.caller.user_id ?? 'anonymous'}</dd></div>
                <div className="flex justify-between gap-3"><dt className="text-soft">Tenant</dt><dd className="font-medium">{i.caller.tenant_id ?? 'none'}</dd></div>
                <div className="flex justify-between gap-3"><dt className="text-soft">Groups</dt><dd className="font-medium">{i.caller.groups.length ? i.caller.groups.join(', ') : 'none'}</dd></div>
                <div className="flex justify-between gap-3"><dt className="text-soft">Role</dt><dd><Badge tone={i.caller.is_admin ? 'accent' : 'neutral'}>{i.caller.is_admin ? 'Administrator' : 'User'}</Badge></dd></div>
                <div className="flex justify-between gap-3"><dt className="text-soft">Authentication</dt><dd><Badge tone={i.config.auth_enabled ? 'ok' : 'warn'}>{i.config.auth_enabled ? 'Required' : 'Off (development)'}</Badge></dd></div>
              </dl>
            ) : <p className="p-5 text-sm text-soft">Not connected.</p>}
          </Card>
          <Card>
            <CardHeader title="About" />
            <div className="space-y-2 p-5 text-sm">
              <p className="text-soft">Server version <span className="font-mono text-ink">{i?.version ?? '–'}</span></p>
              <a className="inline-flex items-center gap-1.5 text-accent hover:underline" href={`${base}/docs`} target="_blank" rel="noreferrer">Interactive API reference <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a>
            </div>
          </Card>
        </div>
      </div>
    </>
  )
}
