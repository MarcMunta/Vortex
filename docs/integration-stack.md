# Vortex Integration Stack

CI full-stack uses CPU-safe stubs for API/control plane and real Vite frontend.

Local:

```powershell
python scripts/integration_stub.py --mode api --port 8000
python scripts/integration_stub.py --mode control --port 8765
cd vortex-chat
npm run test:integration
```

Docker profile:

```powershell
docker compose -f c3_rnt2_ai/docker-compose.integration.yml up --build
```

The stack validates `/v1/status`, `/control/status`, training runs, SSE-compatible endpoints, multimodal status, and frontend boot against real HTTP services without GPU.
