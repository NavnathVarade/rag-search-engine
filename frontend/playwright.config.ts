import { defineConfig, devices } from '@playwright/test'

/** End-to-end tests run against REAL servers (nothing mocked):
 *   E2E_OPEN_URL  - API with AUTH_ENABLED=false serving the built UI (default http://127.0.0.1:8010)
 *   E2E_AUTH_URL  - API with AUTH_ENABLED=true and two API keys (default http://127.0.0.1:8011)
 *   E2E_LLM_URL   - API configured with an LLM provider backed by a local stand-in server (default http://127.0.0.1:8012)
 * See frontend/README.md for the exact commands that start them. */
export const OPEN = process.env.E2E_OPEN_URL ?? 'http://127.0.0.1:8010'
export const AUTH = process.env.E2E_AUTH_URL ?? 'http://127.0.0.1:8011'
export const LLM = process.env.E2E_LLM_URL ?? 'http://127.0.0.1:8012'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: { baseURL: OPEN, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [
    { name: 'desktop', testMatch: /app\.spec\.ts/, use: { ...devices['Desktop Chrome'], viewport: { width: 1360, height: 900 } } },
    { name: 'auth', testMatch: /auth\.spec\.ts/, use: { ...devices['Desktop Chrome'], baseURL: AUTH } },
    { name: 'llm', testMatch: /llm\.spec\.ts/, use: { ...devices['Desktop Chrome'], baseURL: LLM } },
    { name: 'mobile', testMatch: /mobile\.spec\.ts/, use: { ...devices['Pixel 5'] } },
  ],
})
