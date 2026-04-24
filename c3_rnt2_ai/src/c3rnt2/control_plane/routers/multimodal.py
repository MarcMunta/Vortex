from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..dependencies import ControlDependencies
from ..models import ObsidianConfigRequest


def build_multimodal_router(deps: ControlDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/control/voice/status")
    async def control_voice_status() -> dict[str, object]:
        return await asyncio.to_thread(deps.voice_status)

    @router.post("/control/voice/restart")
    async def control_voice_restart() -> dict[str, object]:
        return await asyncio.to_thread(deps.restart_voice)

    @router.get("/control/obsidian/status")
    async def control_obsidian_status() -> dict[str, object]:
        return await asyncio.to_thread(deps.obsidian_status)

    @router.post("/control/obsidian/config")
    async def control_obsidian_config(payload: ObsidianConfigRequest) -> dict[str, object]:
        patch = payload.model_dump(exclude_none=True)
        return await asyncio.to_thread(deps.configure_obsidian, patch)

    @router.get("/control/multimodal/status")
    async def control_multimodal_status() -> dict[str, object]:
        payload = await asyncio.to_thread(deps.multimodal_status)
        return {"ok": True, "status": payload}

    @router.get("/control/multimodal/stream")
    async def control_multimodal_stream() -> StreamingResponse:
        def _events():
            last = ""
            while True:
                payload = {"ts": float(time.time()), "status": deps.multimodal_status()}
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return router
