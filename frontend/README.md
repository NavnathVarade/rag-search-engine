# Docs Intelligence Console (web UI)

React 18 + TypeScript + Vite + Tailwind CSS, with TanStack Query for server state, Radix primitives for dialogs and tabs,
Recharts for charts and react-markdown for answers. It talks only to the real API in this repository: there is no mock data
anywhere in the app.

## Run it

You need the API running first (from the repository root):

```bash
uv sync
uv run uvicorn rag_engine.api.app:app --port 8000          # AUTH_ENABLED=false by default (development)
```

**Option A, production-style (one process).** Build once; the API then serves the UI at <http://localhost:8000/>.

```bash
cd frontend
npm ci
npm run build
# restart or start the API from the repository root so it finds frontend/dist
```

**Option B, development with hot reload.**

```bash
cd frontend
npm ci
npm run dev                                          # http://localhost:5173, proxies /api, /health, /ready, /metrics to :8000
# another API address:  VITE_API_TARGET=http://127.0.0.1:8010 npm run dev
```

**Option C, Docker.** `docker compose up --build` builds the UI inside the API image and serves it at <http://localhost:8000/>.
(The compose files and the Dockerfile frontend stage were checked statically; the image build itself has not been run in this
project's sandbox because it has no Docker daemon.)

## Signing in

With `AUTH_ENABLED=false` no sign-in is needed. With `AUTH_ENABLED=true` the app shows a sign-in screen. Enter an API key
(create one with `rag auth create-key`) or a bearer token. Credentials are kept in `sessionStorage` (this tab only) unless
"Remember on this device" is ticked, and are sent only as `x-api-key` / `Authorization` headers. If the server sets
`METRICS_TOKEN`, enter it under Settings so the Health and Analytics pages can read `/metrics`.

## Pages and the endpoints behind them

| Page | Endpoints used |
|---|---|
| Dashboard | `GET /api/v1/info`, `GET /ready`, `GET /api/v1/documents` |
| Search | `POST /api/v1/search` (modes: hybrid, bm25, tfidf, dense, hybrid_weighted) |
| Ask | `POST /api/v1/chat` |
| Documents | `GET /api/v1/documents`, `GET`/`DELETE /api/v1/documents/{id}` |
| Add content | `POST /api/v1/documents/upload`, `POST /api/v1/documents/ingest-url` |
| Index | `GET /api/v1/info`, `POST /api/v1/index/rebuild` (admin) |
| Analytics | `GET /metrics`, `GET /api/v1/benchmarks` |
| Health | `GET /health`, `GET /ready`, `GET /metrics` |
| Settings | `GET /api/v1/info` |

`/api/v1/info`, `/api/v1/benchmarks`, `GET /api/v1/documents`, `/documents/upload` and `/documents/ingest-url` were added to the
backend for this UI (see `src/rag_engine/api/routes/`). Everything else already existed.

## Checks

```bash
npm run typecheck && npm run lint && npm test      # types, lint, 21 unit tests
npm run build
```

End-to-end tests drive a real browser against real API servers (nothing is mocked). From the repository root:

```bash
cd frontend && npm run build && cd ..
frontend/e2e/servers.sh start                      # :8010 open, :8011 auth + two API keys, :8012 LLM configured (stand-in server on :9100)
cd frontend && npx playwright test                 # 15 tests: desktop, auth/tenant isolation, generated answers, mobile
cd .. && frontend/e2e/servers.sh stop
```

Playwright uses the Chromium that is already installed in this environment; elsewhere run `npx playwright install chromium` once.

## Design notes

- One memorable element: the **score strip** on every search result shows the rank each retriever gave the passage, so keyword
  versus vector contribution is visible at a glance. Values are the API's own, never recomputed.
- Chart colours (blue, then orange) come from a palette validated for colour-blind separation and contrast in light and dark;
  every chart has a table view.
- System fonts only, so the app works offline and under a strict content security policy.
- Limitations the UI states rather than hides: offline stub answers without an LLM, the hashing embedder, one API instance only.
