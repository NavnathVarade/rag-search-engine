import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, ApiError } from '@/lib/api'
import { parsePrometheus } from '@/lib/prometheus'

export const keys = { info: ['info'], docs: ['documents'], doc: (id: string) => ['documents', id], ready: ['ready'], health: ['health'], metrics: ['metrics'], bench: ['benchmarks'] }

export const useInfo = () => useQuery({ queryKey: keys.info, queryFn: api.info, refetchInterval: 30_000 })
export const useDocuments = () => useQuery({ queryKey: keys.docs, queryFn: api.listDocuments })
export const useDocument = (id: string | null) => useQuery({ queryKey: keys.doc(id ?? ''), queryFn: () => api.getDocument(id as string), enabled: !!id })
export const useReady = () => useQuery({ queryKey: keys.ready, queryFn: api.ready, refetchInterval: 15_000 })
export const useHealth = () => useQuery({ queryKey: keys.health, queryFn: api.health, refetchInterval: 15_000 })
export const useBenchmarks = () => useQuery({ queryKey: keys.bench, queryFn: api.benchmarks, staleTime: 5 * 60_000 })
export const useMetrics = () =>
  useQuery({
    queryKey: keys.metrics,
    queryFn: async () => parsePrometheus(await api.metricsText()),
    refetchInterval: 15_000,
    // a 404 means the operator disabled metrics; do not hammer the server
    retry: (n, e) => !(e instanceof ApiError && e.kind !== 'server' && e.kind !== 'network') && n < 2,
  })

/** Refresh everything that depends on the document set. */
export function useInvalidateIndex() {
  const qc = useQueryClient()
  return () => Promise.all([qc.invalidateQueries({ queryKey: keys.docs }), qc.invalidateQueries({ queryKey: keys.info }), qc.invalidateQueries({ queryKey: keys.metrics })])
}

export function useDeleteDocument() {
  const invalidate = useInvalidateIndex()
  return useMutation({ mutationFn: (id: string) => api.deleteDocument(id), onSuccess: invalidate })
}

export function useRebuildIndex() {
  const invalidate = useInvalidateIndex()
  return useMutation({ mutationFn: api.rebuildIndex, onSuccess: invalidate })
}
