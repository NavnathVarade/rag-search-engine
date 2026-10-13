/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'

// In development the UI talks to the API through this proxy, so no CORS setup is needed.
// Override the target with VITE_API_TARGET=http://host:port.
const target = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: {
    port: 5173,
    proxy: { '/api': target, '/health': target, '/ready': target, '/metrics': target },
  },
  build: { sourcemap: false, chunkSizeWarningLimit: 900 },
  test: { environment: 'jsdom', globals: true, setupFiles: ['./src/test/setup.ts'], include: ['src/**/*.test.{ts,tsx}'] },
})
