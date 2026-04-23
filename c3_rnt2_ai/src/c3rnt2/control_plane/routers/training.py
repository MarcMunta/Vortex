from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import TrainingResetRequest, TrainingStartRequest

if TYPE_CHECKING:
    from ...control_server import ControlState


def build_training_router(state: "ControlState") -> APIRouter:
    router = APIRouter()

    @router.post("/control/training/start")
    async def control_training_start(payload: TrainingStartRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            state.start_training,
            payload.mode,
            source=payload.source,
        )

    @router.post("/control/training/reset")
    async def control_training_reset(payload: TrainingResetRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            state.reset_training_state,
            clear_runs=bool(payload.clear_runs),
            clear_learning_queue=bool(payload.clear_learning_queue),
        )

    @router.get("/control/training/runs")
    async def control_training_runs() -> dict[str, Any]:
        runs = await asyncio.to_thread(state.list_runs, include_details=False, limit=120)
        return {"ok": True, "runs": runs}

    @router.get("/control/training/runs/{run_id}")
    async def control_training_run(run_id: str) -> dict[str, Any]:
        payload = await asyncio.to_thread(state.get_run, run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="training_run_not_found")
        return {"ok": True, "run": payload}

    @router.get("/control/training/runs/{run_id}/events")
    async def control_training_run_events(run_id: str) -> dict[str, Any]:
        events = await asyncio.to_thread(state.get_run_events, run_id)
        return {"ok": True, "run_id": run_id, "events": events}

    @router.get("/control/training/runs/{run_id}/logs")
    async def control_training_run_logs(run_id: str) -> dict[str, Any]:
        logs = await asyncio.to_thread(state.get_run_logs, run_id)
        return {"ok": True, "run_id": run_id, "logs": logs}

    @router.get("/control/training/stream")
    async def control_training_stream() -> StreamingResponse:
        def _events():
            last = ""
            while True:
                payload = state._build_training_stream_payload()
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return router
