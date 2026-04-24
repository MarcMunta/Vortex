from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from ..dependencies import ApiDependencies
from ..services import get_api_services


def register_multimodal_routes(app: FastAPI, deps: ApiDependencies) -> None:
    @app.get("/v1/voice/status")
    async def voice_status():
        return JSONResponse(content=get_api_services(app).voice_service.status())

    @app.post("/v1/voice/restart")
    async def voice_restart():
        return JSONResponse(content=get_api_services(app).voice_service.restart())

    @app.get("/v1/voice/audio/{file_name}")
    async def voice_audio(file_name: str):
        target = get_api_services(app).voice_service.resolve_audio_file(file_name)
        if target is None:
            raise HTTPException(
                status_code=404,
                detail=deps.openai_error("voice_audio_not_found", code="not_found"),
            )
        return FileResponse(target)

    @app.post("/v1/voice/transcribe")
    async def voice_transcribe(request: Request):
        content_type = str(request.headers.get("content-type") or "").strip().lower()
        text_hint = None
        raw_audio = None
        language = str(request.headers.get("x-vortex-voice-language") or "").strip() or None
        if content_type.startswith("application/json"):
            payload = await request.json()
            text_hint = str(payload.get("text") or "").strip() or None
            language = str(payload.get("language") or language or "").strip() or None
        else:
            raw_audio = await request.body()
        services = get_api_services(app)
        result = services.voice_service.transcribe(
            raw_audio=raw_audio,
            content_type=content_type,
            text_hint=text_hint,
            language=language,
            session=services.spatial_store.get_session(),
        )
        if not result.get("ok"):
            return JSONResponse(status_code=400, content=result)
        action_result = deps.apply_voice_intent(result.get("intent"))
        if action_result is not None:
            result["action_result"] = action_result
        if result.get("transcript"):
            services.spatial_store.apply_event(
                {
                    "kind": "voice",
                    "transcript": result.get("transcript"),
                    "intent": result.get("intent"),
                }
            )
        return JSONResponse(content=result)

    @app.post("/v1/voice/speak")
    async def voice_speak(request: Request):
        payload = await request.json()
        result = get_api_services(app).voice_service.speak(
            text=str(payload.get("text") or ""),
            language=str(payload.get("language") or "").strip() or None,
        )
        return JSONResponse(content=result, status_code=200 if bool(result.get("ok")) else 400)

    @app.get("/v1/spatial/session")
    async def spatial_session_get():
        return JSONResponse(content={"ok": True, "session": get_api_services(app).spatial_store.get_session()})

    @app.post("/v1/spatial/session")
    async def spatial_session_post(request: Request):
        payload = await request.json()
        session = get_api_services(app).spatial_store.update_session(payload)
        return JSONResponse(content={"ok": True, "session": session})

    @app.post("/v1/spatial/events")
    async def spatial_events(request: Request):
        payload = await request.json()
        session = get_api_services(app).spatial_store.apply_event(payload)
        return JSONResponse(content={"ok": True, "session": session})

    @app.post("/v1/spatial/panels/open")
    async def spatial_panels_open(request: Request):
        payload = await request.json()
        result = get_api_services(app).spatial_store.open_panel(payload)
        return JSONResponse(content=result, status_code=200 if bool(result.get("ok")) else 400)

    @app.post("/v1/spatial/panels/update")
    async def spatial_panels_update(request: Request):
        payload = await request.json()
        panel_id = str(payload.get("panel_id") or payload.get("panelId") or "").strip()
        if not panel_id:
            raise HTTPException(
                status_code=400,
                detail=deps.openai_error("panel_id_required", code="invalid_request"),
            )
        patch = dict(payload)
        patch.pop("panel_id", None)
        patch.pop("panelId", None)
        result = get_api_services(app).spatial_store.update_panel(panel_id, patch)
        return JSONResponse(content=result, status_code=200 if bool(result.get("ok")) else 404)

    @app.post("/v1/spatial/panels/navigate")
    async def spatial_panels_navigate(request: Request):
        payload = await request.json()
        panel_id = str(payload.get("panel_id") or payload.get("panelId") or "").strip()
        if not panel_id:
            raise HTTPException(
                status_code=400,
                detail=deps.openai_error("panel_id_required", code="invalid_request"),
            )
        delta = int(payload.get("delta") or 0)
        index = payload.get("index")
        result = get_api_services(app).spatial_store.navigate_panel(
            panel_id,
            delta=delta if delta else 1,
            index=int(index) if index is not None else None,
        )
        return JSONResponse(content=result, status_code=200 if bool(result.get("ok")) else 404)

    @app.get("/v1/obsidian/status")
    async def obsidian_status():
        return JSONResponse(content=get_api_services(app).obsidian_sync.status())

    @app.post("/v1/obsidian/config")
    async def obsidian_config(request: Request):
        payload = await request.json()
        services = get_api_services(app)
        config = services.obsidian_sync.set_config(payload)
        return JSONResponse(
            content={"ok": True, "config": config, "status": services.obsidian_sync.status()}
        )

    @app.post("/v1/obsidian/save")
    async def obsidian_save(request: Request):
        payload = await request.json()
        result = get_api_services(app).obsidian_sync.save_note(
            note_type=str(payload.get("note_type") or payload.get("type") or "session"),
            title=str(payload.get("title") or "Vortex note"),
            content=str(payload.get("content") or ""),
            metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
            tags=[str(item) for item in (payload.get("tags") or []) if str(item).strip()],
        )
        return JSONResponse(content=result, status_code=200 if bool(result.get("ok")) else 400)
