import { expect, test } from '@playwright/test'
import path from 'node:path'

test('mobile layout: no horizontal scroll, menu drawer navigates, search works', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(0)
  await page.screenshot({ path: path.join(import.meta.dirname, 'shots', '10-mobile-dashboard.png'), fullPage: true })
  await page.getByRole('button', { name: 'Open menu' }).click()
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Search' }).click()
  await expect(page.getByRole('heading', { name: 'Search' })).toBeVisible()
  await page.getByLabel('Search query').fill('restart policy')
  await page.getByRole('button', { name: 'Search' }).click()
  await expect(page.getByText(/\d+ results? in/)).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0)
  await page.screenshot({ path: path.join(import.meta.dirname, 'shots', '11-mobile-search.png'), fullPage: true })
})
