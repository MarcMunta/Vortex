from __future__ import annotations

import argparse
import json
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def _session() -> dict[str, Any]:
    now = time.time()
    return {
        "session_id": "integration",
        "selected_object_id": None,
        "selected_region": None,
        "active_panel_ids": [],
        "active_presentation_id": None,
        "active_page_index": 0,
        "interaction_mode": "inspect",
        "panels": [],
        "created_at": now,
        "updated_at": now,
    }


def _operational_status() -> dict[str, Any]:
    return {
        "ok": True,
        "chat_ready": True,
        "chat_mode": "stub",
        "offline_ready": True,
        "engine_ready": True,
        "model_ready": True,
        "training_ready": True,
        "web_disabled": True,
        "engine_kind": "integration-stub",
        "active_model": "vortex-integration-stub",
    }


def _multimodal_status() -> dict[str, Any]:
    return {
        "ok": True,
        "voice": {"ok": True, "enabled": True, "asr_backend": "stub", "tts_backend": "stub"},
        "spatial": _session(),
        "obsidian": {"ok": True, "enabled": True, "available": True, "validated": True},
        "fusion": {"enabled": True, "summary": "integration stub ready", "refs": []},
    }


def _sse(payload: dict[str, Any]) -> StreamingResponse:
    def _events():
        yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


def build_api_app() -> FastAPI:
    app = FastAPI(title="Vortex API Integration Stub")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "service": "api-stub"}

    @app.get("/readyz")
    @app.get("/v1/status")
    async def status():
        return _operational_status()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "vortex-integration-stub"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        if payload.get("stream"):
            chunk = {
                "id": "stub",
                "choices": [{"delta": {"content": "integration ok"}, "finish_reason": "stop"}],
            }
            return _sse(chunk)
        return {"choices": [{"message": {"role": "assistant", "content": "integration ok"}}]}

    @app.get("/v1/chat/sessions")
    async def sessions_get():
        return {"ok": True, "sessions": []}

    @app.post("/v1/chat/sessions/sync")
    async def sessions_sync(request: Request):
        payload = await request.json()
        sessions = payload.get("sessions") if isinstance(payload, dict) else []
        return {"ok": True, "sessions": sessions if isinstance(sessions, list) else [], "count": len(sessions or [])}

    @app.get("/v1/self-edits/proposals")
    async def proposals():
        return {"ok": True, "data": []}

    @app.get("/v1/voice/status")
    async def voice_status():
        return _multimodal_status()["voice"]

    @app.post("/v1/voice/transcribe")
    async def voice_transcribe():
        return {"ok": True, "transcript": "integration ok", "intent": {"kind": "none"}}

    @app.get("/v1/obsidian/status")
    async def obsidian_status():
        return _multimodal_status()["obsidian"]

    @app.post("/v1/obsidian/config")
    async def obsidian_config(request: Request):
        return {"ok": True, **(await request.json())}

    @app.post("/v1/obsidian/save")
    async def obsidian_save():
        return {"ok": True, "path": "integration.md"}

    @app.get("/v1/spatial/session")
    async def spatial_session_get():
        return {"ok": True, "session": _session()}

    @app.post("/v1/spatial/session")
    @app.post("/v1/spatial/events")
    @app.post("/v1/spatial/panels/open")
    @app.post("/v1/spatial/panels/update")
    @app.post("/v1/spatial/panels/navigate")
    async def spatial_mutate():
        return {"ok": True, "session": _session(), "panel": None}

    return app


def build_control_app() -> FastAPI:
    app = FastAPI(title="Vortex Control Integration Stub")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "service": "control-stub"}

    @app.get("/control/status")
    async def control_status():
        return {
            "ok": True,
            "bootstrap": {"running": False, "stage": "ready", "updated_at": time.time()},
            "docker": {"ready": True, "reason": "integration_stub"},
            "runtime": {"api_ready": True, "runtime_ready": True, "status": _operational_status()},
            "frontend": {"ready": True, "port": 4173, "url": "http://127.0.0.1:4173"},
            "multimodal": _multimodal_status(),
            "autonomy": {"enabled": False, "boot_mode": "manual", "state": "idle", "active_agents": []},
            "runs": [{"run_id": "integration-run", "mode": "quick", "status": "completed", "stage": "done"}],
        }

    @app.get("/control/training/runs")
    async def runs():
        return {"ok": True, "runs": [{"run_id": "integration-run", "mode": "quick", "status": "completed"}]}

    @app.get("/control/training/stream")
    async def training_stream():
        return _sse({"ts": time.time(), "active_run_id": None, "runs": []})

    @app.get("/control/autonomy/status")
    async def autonomy_status():
        return {"ok": True, "autonomy": {"enabled": False, "boot_mode": "manual", "state": "idle", "active_agents": []}}

    @app.get("/control/autonomy/stream")
    async def autonomy_stream():
        return _sse({"ts": time.time(), "status": {"enabled": False, "boot_mode": "manual", "state": "idle", "active_agents": []}, "events": []})

    @app.get("/control/multimodal/status")
    async def multimodal_status():
        return {"ok": True, "status": _multimodal_status()}

    @app.get("/control/multimodal/stream")
    async def multimodal_stream():
        return _sse({"ts": time.time(), "status": _multimodal_status()})

    @app.get("/control/internet/allowlist")
    async def allowlist_get():
        return {"ok": True, "domains": ["react.dev"]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "control"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app = build_api_app() if args.mode == "api" else build_control_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
