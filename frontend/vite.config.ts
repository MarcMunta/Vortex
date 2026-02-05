import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const proxyTarget = env.VITE_API_PROXY_TARGET || env.VITE_API_BASE_URL || 'http://localhost:8000';
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        proxy: {
          '/v1': { target: proxyTarget, changeOrigin: true },
          '/healthz': { target: proxyTarget, changeOrigin: true },
          '/readyz': { target: proxyTarget, changeOrigin: true },
          '/metrics': { target: proxyTarget, changeOrigin: true },
          '/doctor': { target: proxyTarget, changeOrigin: true },
        },
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
