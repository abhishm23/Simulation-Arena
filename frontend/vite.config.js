import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Python backend is a separate process on :8000. Everything the app needs
// lives under /api, so a single proxy entry keeps the browser on one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Server-sent events must not be buffered by the dev proxy.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if ((proxyRes.headers['content-type'] || '').includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform';
              proxyRes.headers['connection'] = 'keep-alive';
              proxyRes.headers['x-accel-buffering'] = 'no';
            }
          });
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
});
