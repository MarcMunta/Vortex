from __future__ import annotations

import importlib
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..dependencies import ApiDependencies


def _probe_nvidia_smi() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or output or f"exit_{proc.returncode}").strip()}
    devices = [line.strip() for line in output.splitlines() if line.strip()]
    return {"ok": bool(devices), "devices": devices}


def _probe_llama_cpp_gpu() -> dict[str, Any]:
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if not callable(fn):
        return {"ok": False, "error": "llama_supports_gpu_offload_unavailable"}
    try:
        return {"ok": bool(fn())}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def register_utility_routes(app: FastAPI, settings: dict, _base_dir: Path, deps: ApiDependencies) -> None:
    @app.get("/doctor")
    @app.post("/doctor")
    async def doctor():
        torch_cuda = bool(deps.torch is not None and deps.torch.cuda.is_available())
        nvidia_smi = _probe_nvidia_smi()
        llama_cpp_gpu = _probe_llama_cpp_gpu()
        payload = {
            "ok": True,
            "profile": str(settings.get("_profile") or ""),
            "backends": list((getattr(app.state, "models", {}) or {}).keys()),
            "training_active": bool(getattr(app.state, "training_active", False)),
            "torch": bool(deps.torch is not None),
            "torch_cuda": torch_cuda,
            "llama_cpp_gpu_offload": bool(llama_cpp_gpu.get("ok")),
            "nvidia_smi": nvidia_smi,
            "cuda": bool(torch_cuda or llama_cpp_gpu.get("ok") or nvidia_smi.get("ok")),
        }
        if not llama_cpp_gpu.get("ok"):
            payload["llama_cpp_gpu_error"] = str(llama_cpp_gpu.get("error") or "")
        if torch_cuda:
            try:
                payload["cuda_device"] = deps.torch.cuda.get_device_name(0)
            except Exception:
                pass
        elif nvidia_smi.get("devices"):
            payload["cuda_device"] = str(nvidia_smi["devices"][0])
        return JSONResponse(content=payload)

    @app.get("/doctor/deep")
    async def doctor_deep():
        payload = {
            "ok": True,
            "profile": str(settings.get("_profile") or ""),
            "backends": list((getattr(app.state, "models", {}) or {}).keys()),
            "deep": True,
            "deep_ok": False,
            "error": None,
        }
        mdl = getattr(app.state, "model", None)
        lock = getattr(app.state, "model_lock", None)
        ctx = lock.read_lock() if lock else nullcontext()
        try:
            with ctx:
                if mdl is None or not hasattr(mdl, "generate"):
                    payload["error"] = "model_missing"
                else:
                    _ = mdl.generate("ping", max_new_tokens=1, temperature=0.0, top_p=1.0)
                    payload["deep_ok"] = True
        except Exception as exc:
            payload["ok"] = False
            payload["error"] = str(exc)
        try:
            cfg = getattr(app.state, "skills_config", None)
            store = getattr(app.state, "skills_store", None)
            strict = bool(getattr(cfg, "strict", True))
            if store is None:
                payload["skills"] = {"ok": True, "skipped": "not_available"}
            else:
                report = store.validate_all(strict=strict)
                staging = []
                try:
                    staging_root = Path(store.staging_root)
                    if staging_root.exists():
                        staging = [p.name for p in staging_root.iterdir() if p.name != ".gitkeep"]
                except Exception:
                    staging = []
                report["staging"] = staging
                if strict and staging:
                    report["ok"] = False
                    report["errors"] = (report.get("errors") or []) + ["staging_not_empty"]
                payload["skills"] = report
        except Exception as exc:
            payload["skills"] = {"ok": False, "error": str(exc)}
        return JSONResponse(content=payload, status_code=200 if payload.get("ok") else 500)

    @app.post("/v1/embeddings")
    async def embeddings():
        raise HTTPException(
            status_code=501,
            detail=deps.openai_error(
                "Embeddings not implemented",
                type="server_error",
                code="not_implemented",
            ),
        )

    @app.get("/v1/files")
    async def list_files():
        return JSONResponse(content={"object": "list", "data": []})

    @app.post("/v1/files")
    async def create_file():
        raise HTTPException(
            status_code=501,
            detail=deps.openai_error(
                "Files not implemented",
                type="server_error",
                code="not_implemented",
            ),
        )

    @app.post("/v1/chat/title")
    async def chat_title(request: Request):
        payload = await request.json()
        message = str(payload.get("message", "")).strip()
        lang = str(payload.get("language", "es")).strip()
        if not message:
            return JSONResponse(status_code=400, content={"ok": False, "error": "message required"})
        import re as _re_title

        title = message
        _prefix_re = _re_title.compile(
            r"^(?:hola|hi|hey|oye|buenas|saludos)[,\s]*",
            flags=_re_title.IGNORECASE,
        )
        _verb_re = _re_title.compile(
            r"^(?:por favor|please|me podr[iÃ­]as?|could you|can you|quiero que|i want you to|dime|tell me|explicame|explain|dame|give me|hazme|muestrame|show me|dar|give|decir|say|necesito|i need)\s+",
            flags=_re_title.IGNORECASE,
        )
        title = _prefix_re.sub("", title).strip()
        title = _verb_re.sub("", title).strip()
        title = _verb_re.sub("", title).strip()
        title = title.rstrip("?!.").strip()
        if title:
            title = title[0].upper() + title[1:]
        if len(title) > 45:
            cut = title[:45].rfind(" ")
            if cut > 20:
                title = title[:cut] + "â€¦"
            else:
                title = title[:45] + "â€¦"
        if not title:
            title = "Chat" if lang == "en" else "ConversaciÃ³n"
        return JSONResponse(content={"ok": True, "title": title})
