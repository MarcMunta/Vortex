from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ..contracts import ChatSessionSyncRequest, ChatSessionsResponse, OperationalStatusResponse
from ..dependencies import ApiDependencies
from ..services import get_api_services


def register_core_routes(app: FastAPI, settings: dict, base_dir, deps: ApiDependencies) -> None:
    def _looks_host_absolute(raw_path: str) -> bool:
        return bool(re.match(r"^[A-Za-z]:[\\/]", raw_path)) or raw_path.startswith("/")

    def _host_mount_candidates(raw_path: str) -> list[Path]:
        raw = str(raw_path or "").strip()
        if not raw:
            return []
        mappings = [
            (
                os.getenv("C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT"),
                os.getenv("C3RNT2_HOST_WORKSPACE_MOUNT"),
            ),
            (
                os.getenv("C3RNT2_HOST_C_WINDOWS_ROOT"),
                os.getenv("C3RNT2_HOST_C_MOUNT"),
            ),
            (
                os.getenv("C3RNT2_HOST_D_WINDOWS_ROOT"),
                os.getenv("C3RNT2_HOST_D_MOUNT"),
            ),
            (
                os.getenv("C3RNT2_HOST_DOWNLOADS_WINDOWS_ROOT"),
                os.getenv("C3RNT2_HOST_DOWNLOADS_MOUNT"),
            ),
        ]
        normalized = [
            (str(host_root or "").strip(), str(mount_root or "").strip())
            for host_root, mount_root in mappings
        ]
        normalized.sort(key=lambda item: len(item[0].rstrip("/\\")), reverse=True)
        candidates: list[Path] = []
        for host_root, mount_root in normalized:
            if not host_root or not mount_root:
                continue
            try:
                host_pure = PureWindowsPath(host_root)
                raw_pure = PureWindowsPath(raw)
                rel = raw_pure.relative_to(host_pure)
            except Exception:
                continue
            mount_path = Path(mount_root)
            if mount_path.exists():
                candidates.append((mount_path / Path(*rel.parts)).resolve())
        return candidates

    def _with_shared_mount_fallback(path: Path, raw_path: str) -> Path:
        if path.exists():
            return path
        for candidate in _host_mount_candidates(raw_path):
            if candidate.exists():
                return candidate
        nested_name = (
            PureWindowsPath(raw_path).name
            if re.match(r"^[A-Za-z]:[\\/]", raw_path)
            else Path(raw_path).name
        )
        repo_name = str(os.getenv("C3RNT2_HOST_WORKSPACE_REPO_NAME") or "").strip()
        base_path = Path(base_dir).resolve()
        if nested_name and nested_name in {base_path.name, repo_name}:
            return base_path
        shared_mount = str(os.getenv("C3RNT2_HOST_WORKSPACE_MOUNT") or "").strip()
        if not shared_mount or not _looks_host_absolute(raw_path):
            return path
        mount_path = Path(shared_mount)
        if not mount_path.exists():
            return path
        host_workspace_root = str(os.getenv("C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT") or "").strip()
        if host_workspace_root:
            try:
                host_pure = PureWindowsPath(host_workspace_root)
                raw_pure = PureWindowsPath(raw_path)
                rel = raw_pure.relative_to(host_pure)
                nested = mount_path / Path(*rel.parts)
                if nested.exists():
                    return nested
            except Exception:
                pass
        if mount_path.name == nested_name:
            return mount_path
        nested = mount_path / nested_name
        return nested

    def _resolve_project_folder(raw: dict) -> Path | None:
        root = str(raw.get("rootPath") or raw.get("root_path") or "").strip()
        project = str(raw.get("projectPath") or raw.get("project_path") or "").strip()
        target = project or root
        if not target:
            return None
        target_path = Path(target)
        if target_path.is_absolute():
            return _with_shared_mount_fallback(target_path, target)
        if root:
            root_path = Path(root)
            if root_path.is_absolute():
                root_clean = root.rstrip("/\\")
                combined_raw = f"{root_clean}/{target}"
                return _with_shared_mount_fallback(root_path / target_path, combined_raw)
        shared_mount = str(os.getenv("C3RNT2_HOST_WORKSPACE_MOUNT") or "").strip()
        if shared_mount:
            shared_candidate = (Path(shared_mount) / target_path).resolve()
            if shared_candidate.exists():
                return shared_candidate
        return _with_shared_mount_fallback((Path(base_dir) / target_path).resolve(), target)

    @app.get("/healthz")
    async def healthz():
        return PlainTextResponse("ok")

    @app.get("/readyz", response_model=OperationalStatusResponse)
    async def readyz():
        payload = deps.build_operational_status(app.state, settings, base_dir)
        if not bool(payload.get("ok", False)):
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(content=payload)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            content=deps.models_list_payload(app.state, settings, base_dir)
        )

    @app.get("/metrics")
    async def metrics():
        services = get_api_services(app)
        text = getattr(services, "metrics", deps.metrics_factory()).render_prometheus()
        sm = getattr(app.state, "skills_metrics", None)
        if sm is not None and hasattr(sm, "render_prometheus"):
            try:
                text += sm.render_prometheus()
            except Exception:
                pass
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    @app.get("/v1/chat/sessions", response_model=ChatSessionsResponse)
    async def list_chat_sessions(account_id: str):
        store = get_api_services(app).chat_sessions_store
        if store is None:
            return JSONResponse(content={"ok": True, "sessions": []})
        account = str(account_id or "").strip()
        if not account:
            raise HTTPException(
                status_code=400,
                detail=deps.openai_error(
                    "account_id_required",
                    type="invalid_request_error",
                    code="account_id_required",
                    param="account_id",
                ),
            )
        return JSONResponse(
            content={"ok": True, "sessions": store.list_sessions(account)}
        )

    @app.post("/v1/chat/sessions/sync", response_model=ChatSessionsResponse)
    async def sync_chat_sessions(request: Request):
        store = get_api_services(app).chat_sessions_store
        if store is None:
            raise HTTPException(
                status_code=501,
                detail=deps.openai_error(
                    "chat_sessions_not_available",
                    type="server_error",
                    code="not_implemented",
                ),
            )
        payload = ChatSessionSyncRequest.model_validate(await request.json())
        account_id = str(payload.account_id or "").strip()
        sessions = payload.sessions
        replace = bool(payload.replace)
        if not account_id:
            raise HTTPException(
                status_code=400,
                detail=deps.openai_error(
                    "account_id_required",
                    type="invalid_request_error",
                    code="account_id_required",
                    param="account_id",
                ),
            )
        synced = store.sync_sessions(account_id, sessions, replace=replace)
        return JSONResponse(
            content={"ok": True, "sessions": synced, "count": len(synced)}
        )

    @app.delete("/v1/chat/sessions")
    async def delete_chat_sessions(account_id: str, session_id: str | None = None):
        store = get_api_services(app).chat_sessions_store
        if store is None:
            return JSONResponse(content={"ok": True})
        account = str(account_id or "").strip()
        target = str(session_id or "").strip()
        if not account:
            raise HTTPException(
                status_code=400,
                detail=deps.openai_error(
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

    @app.post("/v1/workspace/projects/validate")
    async def validate_workspace_projects(request: Request):
        payload = await request.json()
        raw_projects = payload.get("projects") if isinstance(payload, dict) else []
        if not isinstance(raw_projects, list):
            raise HTTPException(
                status_code=400,
                detail=deps.openai_error(
                    "projects_must_be_list",
                    type="invalid_request_error",
                    code="projects_must_be_list",
                    param="projects",
                ),
            )
        results = []
        for raw in raw_projects:
            if not isinstance(raw, dict):
                continue
            project_id = str(raw.get("id") or "").strip()
            folder = _resolve_project_folder(raw)
            valid = bool(folder and folder.exists() and folder.is_dir())
            results.append(
                {
                    "id": project_id,
                    "valid": valid,
                    "path": str(folder) if folder is not None else "",
                }
            )
        return JSONResponse(content={"ok": True, "projects": results})

    @app.post("/v1/workspace/folder-picker")
    async def pick_workspace_folder(request: Request):
        payload = await request.json()
        title = (
            str(payload.get("title") or "").strip()
            if isinstance(payload, dict)
            else ""
        )
        initial_dir = (
            str(payload.get("initial_dir") or payload.get("initialDir") or "").strip()
            if isinstance(payload, dict)
            else ""
        )
        if sys.platform != "win32":
            raise HTTPException(
                status_code=501,
                detail=deps.openai_error(
                    "native_folder_picker_unavailable",
                    type="server_error",
                    code="native_folder_picker_unavailable",
                ),
            )
        description = title or "Selecciona carpeta del proyecto"
        escaped_description = description.replace("'", "''")
        escaped_initial = initial_dir.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::UTF8;"
            "$dlg=New-Object System.Windows.Forms.FolderBrowserDialog;"
            f"$dlg.Description='{escaped_description}';"
            "$dlg.ShowNewFolderButton=$false;"
            f"if('{escaped_initial}' -and (Test-Path -LiteralPath '{escaped_initial}'))"
            "{ $dlg.SelectedPath=(Resolve-Path -LiteralPath '" + escaped_initial + "').Path };"
            "if($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
            "{ Write-Output $dlg.SelectedPath }"
        )
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-STA",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return JSONResponse(content={"ok": True, "cancelled": True, "path": ""})
        if completed.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=deps.openai_error(
                    (completed.stderr or "folder_picker_failed").strip(),
                    type="server_error",
                    code="folder_picker_failed",
                ),
            )
        selected = (completed.stdout or "").strip().splitlines()
        folder = selected[-1].strip() if selected else ""
        return JSONResponse(
            content={"ok": True, "cancelled": not bool(folder), "path": folder}
        )
