from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException

from ..dependencies import ControlDependencies
from ..models import AllowlistRequest, BootstrapRequest


def build_core_router(deps: ControlDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"ok": True, "service": "vortex-control", "ts": float(time.time())}

    @router.get("/control/status")
    async def control_status() -> dict[str, object]:
        return await asyncio.to_thread(deps.status)

    @router.post("/control/bootstrap")
    async def control_bootstrap(payload: BootstrapRequest) -> dict[str, object]:
        return await asyncio.to_thread(
            deps.start_bootstrap,
            force=bool(payload.force),
            mode=payload.mode,
        )

    @router.post("/control/model/init")
    async def control_model_init() -> dict[str, object]:
        return await asyncio.to_thread(
            deps.start_bootstrap,
            force=False,
            mode="ensure",
        )

    @router.post("/control/runtime/restart")
    async def control_restart() -> dict[str, object]:
        return await asyncio.to_thread(deps.restart_runtime)

    @router.post("/control/instructions/reload")
    async def control_reload_instructions() -> dict[str, object]:
        import requests

        resp = requests.post(f"{deps.api_url}/v1/instructions/reload", timeout=10.0)
        payload = resp.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="instructions_reload_invalid")
        return payload

    @router.get("/control/internet/allowlist")
    async def control_allowlist_get() -> dict[str, object]:
        domains = await asyncio.to_thread(deps.get_allowlist)
        return {"ok": True, "domains": domains}

    @router.post("/control/internet/allowlist")
    async def control_allowlist_post(payload: AllowlistRequest) -> dict[str, object]:
        domains = await asyncio.to_thread(deps.set_allowlist, payload.domains)
        return {"ok": True, "domains": domains}

    return router
