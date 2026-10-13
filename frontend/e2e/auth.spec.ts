import { expect, test } from '@playwright/test'

test.describe.configure({ mode: 'serial' })

test('requires sign-in, rejects a bad key, accepts a valid key and keeps it only for the tab', async ({ page, context }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Sign in to the API' })).toBeVisible()
  await page.getByLabel('API key').fill('wrong-key')
  await page.getByRole('button', { name: 'Save and connect' }).click()
  await expect(page.getByRole('heading', { name: 'Sign in to the API' })).toBeVisible()
  await page.getByLabel('API key').fill('e2e-user-key')
  await page.getByRole('button', { name: 'Save and connect' }).click()
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible()
  await expect(page.getByText('alice')).toBeVisible()
  // stored for this tab only, not on disk
  expect(await page.evaluate(() => localStorage.getItem('rag.connection'))).toBeNull()
  expect(await page.evaluate(() => sessionStorage.getItem('rag.connection'))).toContain('e2e-user-key')
  // a new tab has no credentials
  const other = await context.newPage()
  await other.goto('/')
  await expect(other.getByRole('heading', { name: 'Sign in to the API' })).toBeVisible()
})

test('non-admin users cannot rebuild; admin can; metrics need their own token', async ({ page }) => {
  await page.goto('/')
  await page.getByLabel('API key').fill('e2e-user-key')
  await page.getByRole('button', { name: 'Save and connect' }).click()
  await page.goto('/index')
  await expect(page.getByRole('button', { name: 'Rebuild index' })).toBeDisabled()
  await expect(page.getByText('Rebuilding needs an administrator credential')).toBeVisible()
  await page.goto('/status')
  await expect(page.getByText('Metrics need a token')).toBeVisible()

  await page.goto('/settings')
  await page.getByLabel('API key').fill('e2e-admin-key')
  await page.getByLabel('Metrics token (optional)').fill('m3trics')
  await page.getByRole('button', { name: 'Save and connect' }).click()
  await expect(page.getByText('Administrator')).toBeVisible()
  await page.goto('/status')
  await expect(page.getByText('rag_index_documents')).toBeVisible()
})

test('tenant isolation: a private document is invisible to another user', async ({ page, browser }) => {
  // admin1 adds a private document
  await page.goto('/')
  await page.getByLabel('API key').fill('e2e-admin-key')
  await page.getByRole('button', { name: 'Save and connect' }).click()
  await page.goto('/add')
  await page.getByRole('tab', { name: 'Pasted text' }).click()
  await page.getByLabel('Who can see it').selectOption('private')
  await page.getByLabel('Name').fill('secret plan')
  await page.getByLabel('Content').fill('The launch codename is Zephyrwind and must stay confidential.')
  await page.getByRole('button', { name: 'Add text' }).click()
  await expect(page.getByText('Added', { exact: true })).toBeVisible()
  await page.goto('/documents')
  await expect(page.getByText('secret-plan.md').first()).toBeVisible()

  // alice cannot see or search it
  const alice = await browser.newPage({ baseURL: page.url().split('/documents')[0] })
  await alice.goto('/')
  await alice.getByLabel('API key').fill('e2e-user-key')
  await alice.getByRole('button', { name: 'Save and connect' }).click()
  await alice.goto('/documents')
  await expect(alice.getByText('No documents yet')).toBeVisible()
  await alice.goto('/search')
  await alice.getByLabel('Search query').fill('Zephyrwind codename')
  await alice.getByRole('button', { name: 'Search' }).click()
  await expect(alice.getByText('No passages matched')).toBeVisible()
})
