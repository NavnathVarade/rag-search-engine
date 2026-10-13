import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Toaster } from 'sonner'
import { AuthGate } from '@/components/AuthGate'
import { AppShell } from '@/components/layout/AppShell'
import { Skeleton } from '@/components/ui/primitives'
import { ApiError } from '@/lib/api'
import { useTheme } from '@/hooks/useTheme'
import Dashboard from '@/pages/Dashboard'
import Search from '@/pages/Search'
import Chat from '@/pages/Chat'
import Documents from '@/pages/Documents'
import AddContent from '@/pages/AddContent'
import Settings from '@/pages/Settings'
import Health from '@/pages/Health'
import IndexPage from '@/pages/Index'

const Analytics = lazy(() => import('@/pages/Analytics')) // pulls in the charting library

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      refetchOnWindowFocus: false,
      retry: (count, error) => !(error instanceof ApiError && error.kind !== 'server' && error.kind !== 'network') && count < 2,
    },
  },
})

function Themed() {
  const { theme } = useTheme()
  return <Toaster position="bottom-right" theme={theme} richColors closeButton />
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthGate>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<Dashboard />} />
              <Route path="search" element={<Search />} />
              <Route path="chat" element={<Chat />} />
              <Route path="documents" element={<Documents />} />
              <Route path="add" element={<AddContent />} />
              <Route path="index" element={<IndexPage />} />
              <Route path="analytics" element={<Suspense fallback={<Skeleton className="h-64" />}><Analytics /></Suspense>} />
              <Route path="status" element={<Health />} />
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<div className="py-20 text-center text-soft">That page does not exist.</div>} />
            </Route>
          </Routes>
        </AuthGate>
      </BrowserRouter>
      <Themed />
    </QueryClientProvider>
  )
}
