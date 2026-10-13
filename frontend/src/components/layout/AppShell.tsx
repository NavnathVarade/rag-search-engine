import { Activity, BarChart3, BookOpenText, Database, FileStack, LayoutDashboard, Menu, MessageSquareText, Moon, PanelLeftClose, PanelLeftOpen, Search, Settings, Sun, UploadCloud, X } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useTheme } from '@/hooks/useTheme'
import { useInfo, useReady } from '@/hooks/queries'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/primitives'
import { Button } from '@/components/ui/button'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/search', label: 'Search', icon: Search },
  { to: '/chat', label: 'Ask', icon: MessageSquareText },
  { to: '/documents', label: 'Documents', icon: FileStack },
  { to: '/add', label: 'Add content', icon: UploadCloud },
  { to: '/index', label: 'Index', icon: Database },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/status', label: 'Health', icon: Activity },
  { to: '/settings', label: 'Settings', icon: Settings },
]

function Brand({ collapsed }: { collapsed: boolean }) {
  return (
    <div className="flex h-14 items-center gap-2.5 px-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-ink"><BookOpenText className="h-4.5 w-4.5" aria-hidden /></div>
      {!collapsed && <span className="truncate text-[15px] font-semibold tracking-tight">Docs Intelligence</span>}
    </div>
  )
}

function Nav({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="flex-1 space-y-0.5 px-2 py-2">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink key={to} to={to} end={end} onClick={onNavigate} title={collapsed ? label : undefined}
          className={({ isActive }) => cn('flex h-9 items-center gap-3 rounded-md px-2.5 text-sm font-medium transition-colors', isActive ? 'bg-accent-wash text-accent' : 'text-soft hover:bg-raised hover:text-ink', collapsed && 'justify-center px-0')}>
          <Icon className="h-4 w-4 shrink-0" aria-hidden />
          {!collapsed && <span className="truncate">{label}</span>}
          {collapsed && <span className="sr-only">{label}</span>}
        </NavLink>
      ))}
    </nav>
  )
}

function StatusPill() {
  const ready = useReady()
  const info = useInfo()
  if (ready.isLoading) return <Badge>Checking…</Badge>
  if (ready.isError) return <Badge tone="bad">API unreachable</Badge>
  const ok = ready.data?.status === 'ready'
  return (
    <Badge tone={ok ? 'ok' : 'warn'} title={info.data ? `Version ${info.data.version}` : undefined}>
      <span className={cn('h-1.5 w-1.5 rounded-full', ok ? 'bg-ok' : 'bg-warn')} aria-hidden />
      {ok ? 'Ready' : 'Not ready'}
    </Badge>
  )
}

export function AppShell(): ReactNode {
  const { theme, toggle } = useTheme()
  const [collapsed, setCollapsed] = useState(() => { try { return localStorage.getItem('rag.sidebar') === '1' } catch { return false } })
  const [mobileOpen, setMobileOpen] = useState(false)
  const info = useInfo()
  const { pathname } = useLocation()
  useEffect(() => { try { localStorage.setItem('rag.sidebar', collapsed ? '1' : '0') } catch { /* ignore */ } }, [collapsed])
  useEffect(() => { setMobileOpen(false) }, [pathname])
  const current = NAV.find((n) => (n.end ? pathname === n.to : pathname.startsWith(n.to)))?.label ?? ''

  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-accent-ink">Skip to content</a>
      <aside className={cn('fixed inset-y-0 left-0 z-30 hidden flex-col border-r bg-surface transition-[width] duration-200 md:flex', collapsed ? 'w-[60px]' : 'w-60')}>
        <Brand collapsed={collapsed} />
        <Nav collapsed={collapsed} />
        <div className="border-t p-2">
          <Button variant="ghost" size={collapsed ? 'icon' : 'sm'} className={cn(!collapsed && 'w-full justify-start')} onClick={() => setCollapsed((c) => !c)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <><PanelLeftClose className="h-4 w-4" /> Collapse</>}
          </Button>
        </div>
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <button aria-label="Close menu" className="absolute inset-0 bg-ink/40" onClick={() => setMobileOpen(false)} />
          <aside className="relative flex h-full w-64 flex-col bg-surface shadow-xl animate-rise">
            <div className="flex items-center justify-between pr-2"><Brand collapsed={false} /><Button variant="ghost" size="icon" aria-label="Close menu" onClick={() => setMobileOpen(false)}><X className="h-4 w-4" /></Button></div>
            <Nav collapsed={false} onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      <div className={cn('transition-[padding] duration-200', collapsed ? 'md:pl-[60px]' : 'md:pl-60')}>
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-bg/85 px-4 backdrop-blur md:px-8">
          <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open menu" onClick={() => setMobileOpen(true)}><Menu className="h-4 w-4" /></Button>
          <span className="text-sm font-medium text-soft">{current}</span>
          <div className="ml-auto flex items-center gap-2">
            <StatusPill />
            {info.data && (
              <Badge className="hidden sm:inline-flex" title="Signed-in identity">
                {info.data.caller.user_id ?? 'anonymous'}{info.data.caller.is_admin ? ' · admin' : ''}
              </Badge>
            )}
            <Button variant="ghost" size="icon" onClick={toggle} aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </header>
        <main id="main" className="mx-auto w-full max-w-6xl px-4 py-8 md:px-8"><Outlet /></main>
      </div>
    </div>
  )
}
