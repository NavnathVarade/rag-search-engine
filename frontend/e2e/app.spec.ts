import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'

test.describe.configure({ mode: 'serial' })
const fx = (n: string) => path.join(import.meta.dirname, 'fixtures', n)
const shot = (page: Page, name: string) => page.screenshot({ path: path.join(import.meta.dirname, 'shots', `${name}.png`), fullPage: true })
const nav = (page: Page, name: string) => page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name }).click()

test('empty index: dashboard explains the state and the real limitations', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible()
  await expect(page.getByText('Nothing indexed yet')).toBeVisible()
  await expect(page.getByText('Answers use the offline stub')).toBeVisible()
  await expect(page.getByText('Vector search uses a hashing embedder')).toBeVisible()
  await expect(page.getByText('Run a single API instance')).toBeVisible()
  await expect(page.getByText('Ready', { exact: true }).first()).toBeVisible()
  await shot(page, '01-dashboard-empty')
})

test('add content: paste text, upload files, detect duplicates, block private URLs', async ({ page }) => {
  await page.goto('/add')
  // pasted text
  await page.getByRole('tab', { name: 'Pasted text' }).click()
  await page.getByLabel('Name').fill('nginx notes')
  await page.getByLabel('Content').fill('# Nginx\n\nUse proxy_pass to forward requests to an upstream server. Configure upstream blocks for load balancing.')
  await page.getByRole('button', { name: 'Add text' }).click()
  await expect(page.getByText(/Added with \d+ chunks?/).first()).toBeVisible()
  // files
  await page.getByRole('tab', { name: 'Files' }).click()
  await page.getByLabel('Choose files to upload').setInputFiles([fx('redis.md'), fx('kubernetes.md')])
  await expect(page.getByText('Added', { exact: true })).toHaveCount(3)
  // the same file again is reported as a duplicate, not re-added
  await page.getByLabel('Choose files to upload').setInputFiles(fx('redis.md'))
  await expect(page.getByText('Duplicate', { exact: true })).toBeVisible()
  await expect(page.getByText('Already indexed with identical content')).toBeVisible()
  // unsupported type fails with a clear message
  await page.getByLabel('Choose files to upload').setInputFiles({ name: 'tool.exe', mimeType: 'application/octet-stream', buffer: Buffer.from('MZ') })
  await expect(page.getByText(/no parser available for file: tool.exe/).first()).toBeVisible()
  // SSRF protection
  await page.getByRole('tab', { name: 'Web page' }).click()
  await page.getByLabel('Page address').fill('http://127.0.0.1:9/secret')
  await page.getByRole('button', { name: 'Fetch and add' }).click()
  await expect(page.getByText('Failed', { exact: true }).first()).toBeVisible()
  await shot(page, '02-add-content')
})

test('documents: list, filter, inspect chunks, delete with confirmation', async ({ page }) => {
  await page.goto('/documents')
  await expect(page.getByRole('row')).toHaveCount(4) // header + 3 documents
  await page.getByLabel('Filter documents').fill('redis')
  await expect(page.getByRole('row')).toHaveCount(2)
  await page.getByRole('button', { name: 'Inspect redis.md' }).click()
  const drawer = page.getByRole('dialog')
  await expect(drawer.getByText(/append-only file/)).toBeVisible()
  await shot(page, '03-documents-inspect')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Delete redis.md' }).click()
  await page.getByRole('button', { name: 'Cancel' }).click()
  await expect(page.getByRole('row')).toHaveCount(2) // cancel keeps it
  await page.getByRole('button', { name: 'Delete redis.md' }).click()
  await page.getByRole('button', { name: 'Delete document' }).click()
  await expect(page.getByText('No documents match')).toBeVisible()
  await page.getByRole('button', { name: 'Clear filter' }).click()
  await expect(page.getByRole('row')).toHaveCount(3)
  // search no longer finds the deleted document
  await nav(page, 'Search')
  await page.getByLabel('Search query').fill('append-only file AOF')
  await page.getByRole('button', { name: 'Search' }).click()
  await expect(page.getByText(/\d+ results? in|No passages matched/)).toBeVisible()
  // the deleted text must be gone; anything returned is a weak vector-only neighbour with no keyword match
  await expect(page.getByRole('listitem').filter({ hasText: 'append-only' })).toHaveCount(0)
  await expect(page.getByText('Weak match').first()).toBeVisible()
})

test('add redis back, then search shows highlights, rankers and scores for every mode', async ({ page }) => {
  await page.goto('/add')
  await page.getByLabel('Choose files to upload').setInputFiles(fx('redis.md'))
  await expect(page.getByText('Added', { exact: true })).toBeVisible()
  await nav(page, 'Search')
  await page.getByLabel('Search query').fill('kubernetes container restart policy')
  await page.getByRole('button', { name: 'Search' }).click()
  const first = page.getByRole('listitem').filter({ hasText: 'restartPolicy' }).first()
  await expect(first).toBeVisible()
  await expect(first.locator('mark').first()).toBeVisible()
  await expect(first.getByLabel(/Keywords ranked this result number 1/)).toBeVisible()
  await expect(first.getByText(/^rrf /)).toBeVisible()
  await shot(page, '04-search-hybrid')
  for (const mode of ['BM25 keywords', 'TF-IDF keywords', 'Vector similarity', 'Hybrid (weighted)']) {
    await page.getByLabel('Retrieval mode').selectOption({ label: mode })
    await page.getByRole('button', { name: 'Search' }).click()
    await expect(page.getByText(new RegExp(`\\d+ results? in`))).toBeVisible()
  }
  await page.getByLabel('Retrieval mode').selectOption({ label: 'BM25 keywords' })
  await page.getByLabel('Search query').fill('zzzzqqqq')
  await page.getByRole('button', { name: 'Search' }).click()
  await expect(page.getByText('No passages matched')).toBeVisible()
})

test('ask: cited stub answer with expandable sources, and an honest insufficient-evidence reply', async ({ page }) => {
  await page.goto('/chat')
  await page.getByLabel('Your question').fill('What happens when a container fails in Kubernetes?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByText('Offline stub answer.')).toBeVisible()
  const sources = page.getByRole('button', { name: /kubernetes\.md/ })
  await expect(sources.first()).toBeVisible()
  await sources.first().click()
  await expect(page.getByText(/CrashLoopBackOff/).first()).toBeVisible()
  await shot(page, '05-chat-answer')
  await page.getByLabel('Your question').fill('What is the airspeed velocity of an unladen swallow?')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByText('Not enough evidence.')).toBeVisible()
  await shot(page, '06-chat-insufficient')
  await page.getByRole('button', { name: 'Clear conversation' }).click()
  await expect(page.getByText('Try a question')).toBeVisible()
})

test('index: shows real configuration and rebuilds with confirmation', async ({ page }) => {
  await page.goto('/index')
  await expect(page.getByText('BM25 (Okapi)')).toBeVisible()
  await expect(page.getByText('One instance only.')).toBeVisible()
  await page.getByRole('button', { name: 'Rebuild index' }).click()
  await page.getByRole('alertdialog').getByRole('button', { name: 'Rebuild index' }).click()
  await expect(page.getByText(/Index rebuilt: \d+ documents, \d+ chunks/)).toBeVisible()
})

test('analytics: live counters and recorded benchmarks, with a table view', async ({ page }) => {
  await page.goto('/analytics')
  await expect(page.getByText('HTTP requests')).toBeVisible()
  await expect(page.getByText(/Search latency by mode/)).toBeVisible()
  await expect(page.getByText('Ranking quality (nDCG@10)')).toBeVisible()
  await expect(page.getByText(/embedder: hashing/).first()).toBeVisible()
  await page.getByRole('button', { name: 'Table' }).first().click()
  await expect(page.getByRole('cell', { name: 'BM25' }).first()).toBeVisible()
  await page.getByRole('button', { name: 'Chart' }).first().click()
  await shot(page, '07-analytics')
})

test('health: liveness, readiness and parsed Prometheus metrics', async ({ page }) => {
  await page.goto('/status')
  await expect(page.getByText('API process responding')).toBeVisible()
  await expect(page.getByText('Search index built')).toBeVisible()
  await expect(page.getByText('Failing')).toHaveCount(0)
  await expect(page.getByText('rag_index_documents')).toBeVisible()
  await page.getByLabel('Filter metrics').fill('zzz')
  await expect(page.getByText('No matching series')).toBeVisible()
  await shot(page, '08-health')
})

test('settings, theme and sidebar behave and persist', async ({ page }) => {
  await page.goto('/settings')
  await expect(page.getByText('Off (development)')).toBeVisible()
  await expect(page.getByRole('link', { name: /Interactive API reference/ })).toHaveAttribute('href', '/docs')
  const html = page.locator('html')
  const before = await html.evaluate((e) => e.classList.contains('dark'))
  await page.getByRole('button', { name: /Switch to (dark|light) theme/ }).click()
  expect(await html.evaluate((e) => e.classList.contains('dark'))).toBe(!before)
  await shot(page, '09-settings-' + (before ? 'light' : 'dark'))
  await page.reload()
  expect(await html.evaluate((e) => e.classList.contains('dark'))).toBe(!before)
  await page.getByRole('button', { name: 'Collapse sidebar' }).click()
  await page.reload()
  await expect(page.getByRole('button', { name: 'Expand sidebar' })).toBeVisible()
})

test('unknown client route shows a message and the API still wins for /api paths', async ({ page, request }) => {
  await page.goto('/nope/nothing')
  await expect(page.getByText('That page does not exist.')).toBeVisible()
  const r = await request.get('/api/v1/does-not-exist')
  expect(r.status()).toBe(404)
  expect((await r.json()).error).toBe('not_found')
})
