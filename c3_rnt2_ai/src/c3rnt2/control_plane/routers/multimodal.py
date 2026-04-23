from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models import ObsidianConfigRequest

if TYPE_CHECKING:
    from ...control_server import ControlState


def build_multimodal_router(state: "ControlState") -> APIRouter:
    router = APIRouter()

    @router.get("/control/voice/status")
    async def control_voice_status() -> dict[str, Any]:
        return await asyncio.to_thread(state.voice_status)

    @router.post("/control/voice/restart")
    async def control_voice_restart() -> dict[str, Any]:
        return await asyncio.to_thread(state.restart_voice)

    @router.get("/control/obsidian/status")
    async def control_obsidian_status() -> dict[str, Any]:
        return await asyncio.to_thread(state.obsidian_status)

    @router.post("/control/obsidian/config")
    async def control_obsidian_config(payload: ObsidianConfigRequest) -> dict[str, Any]:
        patch = payload.model_dump(exclude_none=True)
        return await asyncio.to_thread(state.configure_obsidian, patch)

    @router.get("/control/multimodal/status")
    async def control_multimodal_status() -> dict[str, Any]:
        payload = await asyncio.to_thread(state.multimodal_status)
        return {"ok": True, "status": payload}

    @router.get("/control/multimodal/stream")
    async def control_multimodal_stream() -> StreamingResponse:
        def _events():
            last = ""
            while True:
                payload = {"ts": float(time.time()), "status": state.multimodal_status()}
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return router
