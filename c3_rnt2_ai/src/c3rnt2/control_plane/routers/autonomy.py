from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models import AutonomyConfigRequest

if TYPE_CHECKING:
    from ...control_server import ControlState


def build_autonomy_router(state: "ControlState") -> APIRouter:
    router = APIRouter()

    @router.get("/control/autonomy/status")
    async def control_autonomy_status() -> dict[str, Any]:
        runtime = await asyncio.to_thread(state.runtime_status)
        runs = await asyncio.to_thread(state.list_runs, include_details=False, limit=12)
        autonomy = await asyncio.to_thread(state.autonomy_status, runtime=runtime, runs=runs)
        return {"ok": True, "autonomy": autonomy}

    @router.post("/control/autonomy/start")
    async def control_autonomy_start() -> dict[str, Any]:
        return await asyncio.to_thread(state.start_autonomy)

    @router.post("/control/autonomy/stop")
    async def control_autonomy_stop() -> dict[str, Any]:
        return await asyncio.to_thread(state.stop_autonomy)

    @router.post("/control/autonomy/config")
    async def control_autonomy_config(payload: AutonomyConfigRequest) -> dict[str, Any]:
        return await asyncio.to_thread(state.configure_autonomy, payload)

    @router.get("/control/autonomy/stream")
    async def control_autonomy_stream() -> StreamingResponse:
        def _events():
            last = ""
            while True:
                runtime = state.runtime_status()
                runs = state.list_runs(include_details=False, limit=6)
                payload = {
                    "ts": float(time.time()),
                    "status": state.autonomy_status(runtime=runtime, runs=runs),
                    "events": state._latest_autonomy_events(limit=16),
                    "active_run_id": state._active_run_id,
                    "runs": runs,
                }
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return router
