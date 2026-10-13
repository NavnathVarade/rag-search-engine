import { useSyncExternalStore } from 'react'

/** Connection settings and credentials.
 *  Credentials live in sessionStorage (cleared when the tab closes) unless the user ticks "remember",
 *  in which case they are kept in localStorage on this device. They are only ever sent as request headers. */
export type CredentialKind = 'api-key' | 'bearer'
export interface Connection {
  baseUrl: string // '' = same origin (or the Vite dev proxy)
  credentialKind: CredentialKind
  credential: string
  metricsToken: string
  remember: boolean
}

const KEY = 'rag.connection'
const EMPTY: Connection = { baseUrl: '', credentialKind: 'api-key', credential: '', metricsToken: '', remember: false }

function read(): Connection {
  try {
    const raw = localStorage.getItem(KEY) ?? sessionStorage.getItem(KEY)
    if (raw) return { ...EMPTY, ...(JSON.parse(raw) as Partial<Connection>) }
  } catch {
    /* storage blocked or corrupt: fall through to defaults */
  }
  return EMPTY
}

let current = read()
const listeners = new Set<() => void>()

export function getConnection(): Connection {
  return current
}

export function saveConnection(next: Connection): void {
  current = { ...next, baseUrl: next.baseUrl.trim().replace(/\/+$/, '') }
  try {
    localStorage.removeItem(KEY)
    sessionStorage.removeItem(KEY)
    ;(current.remember ? localStorage : sessionStorage).setItem(KEY, JSON.stringify(current))
  } catch {
    /* keep the in-memory value for this page load */
  }
  listeners.forEach((l) => l())
}

export function clearCredentials(): void {
  saveConnection({ ...current, credential: '', metricsToken: '' })
}

export function authHeaders(c: Connection = current): Record<string, string> {
  if (!c.credential) return {}
  return c.credentialKind === 'bearer' ? { Authorization: `Bearer ${c.credential}` } : { 'x-api-key': c.credential }
}

export function useConnection(): Connection {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
    () => current,
  )
}
