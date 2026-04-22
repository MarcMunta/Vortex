# Vortex Chat (frontend)

Frontend Vite + React para el backend local **Vortex** (`/v1/*`, estilo OpenAI, streaming SSE).

- Nunca llama a proveedores LLM desde el navegador.
- En dev usa el proxy de Vite (`vite.config.ts`) para hablar con el backend y, si corresponde, inyectar `Authorization`.

## Ejecutar (recomendado)

Desde la raíz del repo:

```powershell
.\run.bat
```

Arranca backend + frontend y abre Chrome en `http://localhost:5173`.

## Ejecutar manual

1) Backend:

```powershell
cd ..\c3_rnt2_ai
python -m vortex serve --host 0.0.0.0 --port 8000
```

2) Frontend:

```powershell
cd ..\vortex-chat
npm install
copy .env.local.example .env.local
npm run dev
```

## Variables de entorno (opcional)

- `VORTEX_BACKEND_PORT` (o `BACKEND_PORT`): puerto del backend (default `8000`).
- `VORTEX_API_TOKEN` (o `KLIMEAI_API_TOKEN`): token Bearer si el backend lo requiere para `/v1/*` (se inyecta desde el proxy; no se expone al navegador).
- `VITE_CONTROL_BASE_URL`: opcional, solo si quieres forzar una URL externa para `/control/*`. Por defecto la UI usa mismo origen y, en dev, el proxy de Vite.

## Docker

```powershell
..\run_docker.ps1
```

El frontend queda servido en `http://127.0.0.1:4173` y hace reverse proxy a:
- backend API (`/v1`, `/doctor`, `/metrics`, `/readyz`, `/healthz`)
- control plane (`/control/*`)
