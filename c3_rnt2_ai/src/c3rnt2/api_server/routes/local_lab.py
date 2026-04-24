from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..dependencies import ApiDependencies


def register_local_lab_routes(app: FastAPI, settings: dict, base_dir: Path, deps: ApiDependencies) -> None:
    @app.get("/v1/local-lab/status")
    async def local_lab_status():
        return JSONResponse(content=deps.collect_local_lab_status(settings, base_dir))

    @app.post("/v1/local-lab/init")
    async def local_lab_init():
        return JSONResponse(content=deps.ensure_host_layout(settings, base_dir))

    @app.get("/v1/local-lab/modules")
    async def local_lab_modules():
        return JSONResponse(content={"object": "list", "data": deps.list_modules(settings, base_dir)})

    @app.get("/v1/local-lab/progress")
    async def local_lab_progress():
        return JSONResponse(content=deps.load_progress(settings, base_dir))

    @app.get("/v1/local-lab/next")
    async def local_lab_next():
        return JSONResponse(content=deps.next_module(settings, base_dir))

    @app.get("/v1/local-lab/roadmap")
    async def local_lab_roadmap():
        return JSONResponse(content=deps.write_roadmap(settings, base_dir))

    @app.get("/v1/local-lab/bootstrap-plan")
    async def local_lab_bootstrap_plan():
        return JSONResponse(content=deps.write_bootstrap_plan(settings, base_dir))

    @app.get("/v1/local-lab/rag-sources")
    async def local_lab_rag_sources():
        return JSONResponse(content=deps.write_rag_sources_manifest(settings, base_dir))

    @app.post("/v1/local-lab/lessons")
    async def local_lab_lessons(request: Request):
        payload = await request.json()
        module_id = str(payload.get("module_id") or "").strip()
        workspace_root = payload.get("workspace_root")
        if not module_id:
            raise HTTPException(status_code=400, detail="module_id required")
        try:
            result = deps.create_lesson(
                settings,
                base_dir,
                module_id=module_id,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(content=result)

    @app.post("/v1/local-lab/check")
    async def local_lab_check(request: Request):
        payload = await request.json()
        workspace = str(payload.get("workspace") or "").strip()
        if not workspace:
            raise HTTPException(status_code=400, detail="workspace required")
        try:
            result = deps.check_lesson(settings, base_dir, workspace=workspace)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(content=result)

    @app.get("/v1/autolearn/status")
    async def autolearn_status():
        try:
            from ...autolearn import _load_state as _al_load_state

            state = _al_load_state(base_dir)
            return JSONResponse(content={"ok": True, **state})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    @app.post("/v1/autolearn/trigger")
    async def autolearn_trigger():
        try:
            from ...autolearn import run_autolearn_tick

            model_ref = app.state.model
            tokenizer_ref = getattr(model_ref, "tokenizer", None)
            if tokenizer_ref is None:
                tokenizer_ref = getattr(model_ref, "_tokenizer", None)
            result = run_autolearn_tick(
                base_dir,
                settings,
                model=model_ref,
                tokenizer=tokenizer_ref,
                force=True,
            )
            return JSONResponse(content=result)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})
