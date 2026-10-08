import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the FastAPI backend runs on :8000 (uvicorn api.server:app).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
