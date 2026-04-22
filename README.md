# Vortex (API + UI local, Windows-first)

Monorepo con:
- `c3_rnt2_ai/`: backend **Vortex** (FastAPI) con endpoints estilo OpenAI (`/v1/*`) + streaming SSE.
- `vortex-chat/`: UI **Vortex** (Vite + React) que llama al backend local (nunca a proveedores LLM desde el navegador).

## Quickstart Windows (RTX 4080 / 64GB)

### Requisitos
- **Python 3.10+**
- **Node.js + npm**
- **Git** (requerido para aplicar parches con rollback)
- (Opcional) CUDA / drivers si usas GPU

### Variables de entorno (mínimas)
- Auth API (opcional, recomendado):
  - `VORTEX_API_TOKEN=devtoken` (compat: `KLIMEAI_API_TOKEN=devtoken`)
- Puertos (opcionales):
  - `VORTEX_BACKEND_PORT=8000`
  - `VORTEX_FRONTEND_PORT=5173`
- Toggles (opcionales):
  - `ENABLE_SELF_TRAIN=1` (self-train loop seguro)
  - `ENABLE_AUTO_EDITS=1` (watcher de propuestas de auto-edición, seguro)

### One command run
Arranca **backend + frontend** (y opcionalmente self-train / auto-edits):

```powershell
.\run.bat
```

Comandos útiles:
```powershell
.\status.bat
.\logs.bat backend
.\logs.bat frontend
.\stop.bat
```

Abrir UI:
- `http://localhost:5173` (por defecto con `run.bat`; abre automÃ¡ticamente en Chrome si estÃ¡ instalado)

### Docker local
Arranque unificado en Docker:

```powershell
.\run_docker.ps1
```

Ese wrapper ejecuta `docker compose -f .\c3_rnt2_ai\docker-compose.yml up -d vortex-api`, y al levantar `vortex-api` arrastra tambiÃ©n:
- `sglang-runtime`
- `vortex-control`
- `vortex-frontend`

URLs:
- `http://127.0.0.1:4173` frontend
- `http://127.0.0.1:8765/control/status` control/status
- `http://127.0.0.1:8000/readyz` backend

Parar stack Docker:

```powershell
.\stop_docker.ps1
```

### Doctor / Bench (opcional)
Perfil recomendado (4080 safe): `C3RNT2_PROFILE=rtx4080_16gb_safe`.

```powershell
cd c3_rnt2_ai
.\.venv\Scripts\python.exe -m vortex doctor --profile $env:C3RNT2_PROFILE
.\.venv\Scripts\python.exe -m vortex doctor --deep --mock --profile $env:C3RNT2_PROFILE
.\.venv\Scripts\python.exe -m vortex bench --profile $env:C3RNT2_PROFILE --max-new 64
```

## Arquitectura (simple)

```
[Frontend (Vite/React)]  --->  HTTP  --->  [Backend API (FastAPI)]
                                   |
                                   +--> (opt) Self-train loop (safe)
                                   +--> (opt) Auto-edits watcher (safe, proposals only)
                                   +--> /metrics + /doctor
```

## “Safe Self-Edit Model” (OBLIGATORIO)

Regla: **prohibido modificar el repo “en caliente” sin aprobación humana**.

Pipeline:
1) `proposal` (se guarda en disco)  
2) `accept/reject` (aprobación humana)  
3) `apply` (aplica el patch + validación)  
4) `rollback` automático si falla la validación

Detalles:
- Propuestas guardadas en `c3_rnt2_ai/skills/_proposals/self_edits/<id>/` (ej: `meta.json`, `patch.diff`, `apply.json`).
- API: `GET /v1/self-edits/proposals?status=pending`, `POST .../{id}/accept|reject|apply`.
- Si `KLIMEAI_API_TOKEN` / `VORTEX_API_TOKEN` está definido, los endpoints `/v1/*` requieren `Authorization: Bearer ...`.
- `apply` ejecuta validación mínima y hace rollback si falla:
  - `pytest -q` (repo root)
  - `python -m c3rnt2 skills validate --all`
  - `python -m c3rnt2 doctor --deep --mock --profile <perfil>`
  - si el patch toca `vortex-chat/`, también `npm run build` (best-effort)
- `apply` requiere **working tree limpio** (`git status` sin cambios). `logs/` y `.pids/` están ignorados por git.

## Frontend: “Personal AI”

En la UI hay un panel anclado **Personal AI** con:
- Chat personal (historial separado en localStorage)
- Tab **Auto-ediciones** con propuestas (píldoras + explorador de diffs)
- Botones **Aceptar / Rechazar / Aplicar** por propuesta
- Badge rojo cuando hay propuestas pendientes

Tip: puedes crear una propuesta demo desde el panel **Auto-ediciones** (botón “Demo”) o arrancar el watcher con `ENABLE_AUTO_EDITS=1`.

## Backend (manual)

```powershell
cd c3_rnt2_ai
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[api]"
.\.venv\Scripts\python.exe -m vortex serve --host 0.0.0.0 --port 8000
```

Validación rápida:
```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"core\",\"messages\":[{\"role\":\"user\",\"content\":\"hola\"}]}"
```

## Troubleshooting

- **Puerto ocupado**: cambia `VORTEX_BACKEND_PORT` / `VORTEX_FRONTEND_PORT` y reintenta.
- **Fallo de deps Python**: borra `.venv/` y ejecuta `.\run.bat` de nuevo.
- **Fallo de deps frontend**: borra `vortex-chat/node_modules/` y ejecuta `.\run.bat`.
- **401 Unauthorized**: define `VORTEX_API_TOKEN` (o `KLIMEAI_API_TOKEN`) y reintenta con `.\run.bat` (el frontend proxya con auth automÃ¡tica).
