from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..dependencies import ControlDependencies
from ..models import AutonomyConfigRequest


def build_autonomy_router(deps: ControlDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/control/autonomy/status")
    async def control_autonomy_status() -> dict[str, object]:
        runtime = await asyncio.to_thread(deps.runtime_status)
        runs = await asyncio.to_thread(deps.list_runs, include_details=False, limit=12)
        autonomy = await asyncio.to_thread(deps.autonomy_status, runtime=runtime, runs=runs)
        return {"ok": True, "autonomy": autonomy}

    @router.post("/control/autonomy/start")
    async def control_autonomy_start() -> dict[str, object]:
        return await asyncio.to_thread(deps.start_autonomy)

    @router.post("/control/autonomy/stop")
    async def control_autonomy_stop() -> dict[str, object]:
        return await asyncio.to_thread(deps.stop_autonomy)

    @router.post("/control/autonomy/config")
    async def control_autonomy_config(payload: AutonomyConfigRequest) -> dict[str, object]:
        return await asyncio.to_thread(deps.configure_autonomy, payload)

    @router.get("/control/autonomy/stream")
    async def control_autonomy_stream() -> StreamingResponse:
        def _events():
            last = ""
            while True:
                runtime = deps.runtime_status()
                runs = deps.list_runs(include_details=False, limit=6)
                payload = {
                    "ts": float(time.time()),
                    "status": deps.autonomy_status(runtime=runtime, runs=runs),
                    "events": deps.latest_autonomy_events(limit=16),
                    "active_run_id": deps.state._active_run_id,
                    "runs": runs,
                }
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return router
