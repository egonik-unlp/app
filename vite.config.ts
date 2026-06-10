import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// For HMR-driven UI work run `wrangler dev` (serves /api on :8787) alongside
// `npm run dev`; the proxy below forwards API calls to it. For an integrated
// run just use `npm run cf:dev`.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8787',
    },
  },
})
