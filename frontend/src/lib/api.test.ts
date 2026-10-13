import { api, ApiError, describeError } from './api'
import { saveConnection } from './auth'

const reply = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers })

describe('api client (transport only; fetch is stubbed here, never in the running app)', () => {
  afterEach(() => { vi.unstubAllGlobals(); saveConnection({ baseUrl: '', credentialKind: 'api-key', credential: '', metricsToken: '', remember: false }) })

  it('sends the API key header and the configured base URL', async () => {
    saveConnection({ baseUrl: 'http://api.test/', credentialKind: 'api-key', credential: 'k1', metricsToken: '', remember: false })
    const f = vi.fn().mockResolvedValue(reply(200, { status: 'ok' }))
    vi.stubGlobal('fetch', f)
    await api.health()
    expect(f.mock.calls[0][0]).toBe('http://api.test/health')
    expect(f.mock.calls[0][1].headers).toMatchObject({ 'x-api-key': 'k1' })
  })

  it('sends bearer tokens as Authorization', async () => {
    saveConnection({ baseUrl: '', credentialKind: 'bearer', credential: 'jwt.x.y', metricsToken: '', remember: false })
    const f = vi.fn().mockResolvedValue(reply(200, {}))
    vi.stubGlobal('fetch', f)
    await api.info()
    expect(f.mock.calls[0][1].headers).toMatchObject({ Authorization: 'Bearer jwt.x.y' })
  })

  it('maps the backend error envelope, status kind and Retry-After', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply(429, { error: 'rate_limited', message: 'slow down' }, { 'Retry-After': '12' })))
    const e = await api.info().catch((x) => x)
    expect(e).toBeInstanceOf(ApiError)
    expect(e).toMatchObject({ status: 429, code: 'rate_limited', kind: 'rate_limited', retryAfter: 12 })
    expect(describeError(e)).toContain('12 seconds')
  })

  it('maps 401 and 503 persistence failures to actionable messages', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply(401, { error: 'unauthorized', message: 'x' })))
    expect(describeError(await api.info().catch((x) => x))).toMatch(/Settings/)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply(503, { error: 'persistence_failed', message: 'x' })))
    expect(describeError(await api.deleteDocument('d').catch((x) => x))).toMatch(/nothing was changed/)
  })

  it('reports FastAPI validation errors and non-JSON failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply(422, { detail: [{ msg: 'String should have at least 1 character' }] })))
    expect((await api.search('', 'hybrid', 5).catch((x) => x)).message).toMatch(/at least 1 character/)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply(502, '<html>bad gateway</html>')))
    expect(await api.info().catch((x) => x)).toMatchObject({ status: 502, kind: 'server' })
  })

  it('turns a network failure into a network ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const e = await api.info().catch((x) => x)
    expect(e.kind).toBe('network')
    expect(describeError(e)).toMatch(/Cannot reach the API/)
  })

  it('reads /ready even when it answers 503', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(reply(503, { status: 'not_ready', checks: { database: false } })))
    expect((await api.ready()).checks.database).toBe(false)
  })

  it('uploads as multipart with a visibility field', async () => {
    const f = vi.fn().mockResolvedValue(reply(200, { document_id: 'd', was_duplicate: false, chunk_count: 1, stage_timings_ms: {} }))
    vi.stubGlobal('fetch', f)
    await api.upload(new Blob(['hi']), 'a.md', 'private')
    const body = f.mock.calls[0][1].body as FormData
    expect(body.get('visibility')).toBe('private')
    expect((body.get('file') as File).name).toBe('a.md')
  })
})
