import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  const resolveEnv = (key: string): string | undefined => {
    const raw = process.env[key] ?? env[key];
    const val = String(raw ?? '').trim();
    return val ? val : undefined;
  };

  const backendPort = resolveEnv('VORTEX_BACKEND_PORT') || resolveEnv('BACKEND_PORT') || '8000';
  const controlBaseUrl = resolveEnv('VITE_CONTROL_BASE_URL');
  const controlPort = resolveEnv('VORTEX_CONTROL_PORT') || resolveEnv('CONTROL_PORT') || resolveEnv('VITE_CONTROL_PORT') || '8765';
  const apiToken = resolveEnv('VORTEX_API_TOKEN') || resolveEnv('KLIMEAI_API_TOKEN');
  const target = `http://127.0.0.1:${backendPort}`;
  const controlTarget = controlBaseUrl || `http://127.0.0.1:${controlPort}`;

  const maybeAttachAuth = (proxy: any) => {
    if (!apiToken) return;
    proxy.on('proxyReq', (proxyReq: any) => {
      proxyReq.setHeader('Authorization', `Bearer ${apiToken}`);
    });
  };

  return {
    server: {
      port: 5173,
      host: '0.0.0.0',
      proxy: {
        '/v1': { target, changeOrigin: true, secure: false, configure: maybeAttachAuth },
        '/doctor': { target, changeOrigin: true, secure: false, configure: maybeAttachAuth },
        '/metrics': { target, changeOrigin: true, secure: false, configure: maybeAttachAuth },
        '/readyz': { target, changeOrigin: true, secure: false, configure: maybeAttachAuth },
        '/healthz': { target, changeOrigin: true, secure: false },
        '/control': { target: controlTarget, changeOrigin: true, secure: false },
      },
    },
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
  };
});
