from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse


def register_core_routes(app: Any, settings: dict, base_dir) -> None:
    from ... import server as legacy

    @app.get("/healthz")
    async def healthz():
        return PlainTextResponse("ok")

    @app.get("/readyz")
    async def readyz():
        payload = legacy._build_operational_status(app.state, settings, base_dir)
        if not bool(payload.get("ok", False)):
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(content=payload)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            content=legacy._models_list_payload(app.state, settings, base_dir)
        )

    @app.get("/metrics")
    async def metrics():
        text = getattr(app.state, "metrics", legacy._MetricsState()).render_prometheus()
        sm = getattr(app.state, "skills_metrics", None)
        if sm is not None and hasattr(sm, "render_prometheus"):
            try:
                text += sm.render_prometheus()
            except Exception:
                pass
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    @app.get("/v1/chat/sessions")
    async def list_chat_sessions(account_id: str):
        store = getattr(app.state, "chat_sessions_store", None)
        if store is None:
            return JSONResponse(content={"ok": True, "sessions": []})
        account = str(account_id or "").strip()
        if not account:
            raise HTTPException(
                status_code=400,
                detail=legacy._openai_error(
                    "account_id_required",
                    type="invalid_request_error",
                    code="account_id_required",
                    param="account_id",
                ),
            )
        return JSONResponse(
            content={"ok": True, "sessions": store.list_sessions(account)}
        )

    @app.post("/v1/chat/sessions/sync")
    async def sync_chat_sessions(request: Request):
        store = getattr(app.state, "chat_sessions_store", None)
        if store is None:
            raise HTTPException(
                status_code=501,
                detail=legacy._openai_error(
                    "chat_sessions_not_available",
                    type="server_error",
                    code="not_implemented",
                ),
            )
        payload = await request.json()
        account_id = str(payload.get("account_id") or "").strip()
        sessions = payload.get("sessions")
        replace = bool(payload.get("replace", True))
        if not account_id:
            raise HTTPException(
                status_code=400,
                detail=legacy._openai_error(
                    "account_id_required",
                    type="invalid_request_error",
                    code="account_id_required",
                    param="account_id",
                ),
            )
        if not isinstance(sessions, list):
            raise HTTPException(
                status_code=400,
                detail=legacy._openai_error(
                    "sessions_required",
                    type="invalid_request_error",
                    code="sessions_required",
                    param="sessions",
                ),
            )
        synced = store.sync_sessions(account_id, sessions, replace=replace)
        return JSONResponse(
            content={"ok": True, "sessions": synced, "count": len(synced)}
        )

    @app.delete("/v1/chat/sessions")
    async def delete_chat_sessions(account_id: str, session_id: str | None = None):
        store = getattr(app.state, "chat_sessions_store", None)
        if store is None:
            return JSONResponse(content={"ok": True})
        account = str(account_id or "").strip()
        target = str(session_id or "").strip()
        if not account:
            raise HTTPException(
                status_code=400,
                detail=legacy._openai_error(
                    "account_id_required",
                    type="invalid_request_error",
                    code="account_id_required",
                    param="account_id",
                ),
            )
        if target:
            store.delete_session(account, target)
        else:
            store.clear_account(account)
        return JSONResponse(content={"ok": True})
