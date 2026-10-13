import { authHeaders, getConnection, saveConnection } from './auth'

const base = { baseUrl: '', credentialKind: 'api-key' as const, credential: 'secret', metricsToken: '', remember: false }

describe('credential storage', () => {
  beforeEach(() => { localStorage.clear(); sessionStorage.clear() })

  it('keeps credentials in sessionStorage by default, not localStorage', () => {
    saveConnection(base)
    expect(sessionStorage.getItem('rag.connection')).toContain('secret')
    expect(localStorage.getItem('rag.connection')).toBeNull()
  })

  it('moves them to localStorage only when "remember" is chosen, and never both', () => {
    saveConnection({ ...base, remember: true })
    expect(localStorage.getItem('rag.connection')).toContain('secret')
    expect(sessionStorage.getItem('rag.connection')).toBeNull()
    saveConnection(base)
    expect(localStorage.getItem('rag.connection')).toBeNull()
  })

  it('trims trailing slashes from the base URL and builds headers by kind', () => {
    saveConnection({ ...base, baseUrl: 'http://x:8000///' })
    expect(getConnection().baseUrl).toBe('http://x:8000')
    expect(authHeaders()).toEqual({ 'x-api-key': 'secret' })
    expect(authHeaders({ ...base, credentialKind: 'bearer' })).toEqual({ Authorization: 'Bearer secret' })
    expect(authHeaders({ ...base, credential: '' })).toEqual({})
  })
})
