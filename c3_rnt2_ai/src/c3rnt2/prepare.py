from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .config import resolve_web_allowlist, resolve_web_strict
from .model_init import DEFAULT_MODEL_ID, model_cache_status, resolve_cache_dir


def _truthy(val: object) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _resolve_path(base_dir: Path, raw: object | None) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _is_local_base_url(raw: object | None) -> bool:
    if raw is None:
        return False
    try:
        parsed = urlparse(str(raw).strip())
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    if host in {
        "127.0.0.1",
        "localhost",
        "::1",
        "host.docker.internal",
        "gateway.docker.internal",
    }:
        return True
    if _truthy(os.getenv("C3RNT2_ASSUME_DOCKER_READY")):
        if host.endswith(".docker.internal"):
            return True
        if host and "." not in host:
            return True
    return False


def _http_json(url: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _docker_ready_status(settings: dict, *, base_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    docker_cfg = settings.get("docker", {}) or {}
    compose_path = _resolve_path(base_dir, docker_cfg.get("compose_path"))
    meta: dict[str, Any] = {"compose_path": str(compose_path) if compose_path else None}
    if not bool(docker_cfg.get("enabled", False)):
        return True, "docker_not_required", meta
    if _truthy(os.getenv("C3RNT2_ASSUME_DOCKER_READY")):
        meta["assumed_from_env"] = True
        return True, "docker_managed_by_host", meta
    if compose_path is None:
        return False, "docker_compose_path_missing", meta
    if not compose_path.exists():
        return False, "docker_compose_missing", meta
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except FileNotFoundError:
        return False, "docker_not_installed", meta
    except Exception as exc:
        meta["detail"] = str(exc)
        return False, "docker_unavailable", meta
    if result.returncode != 0:
        meta["detail"] = (result.stderr or result.stdout or "").strip() or None
        return False, "docker_unavailable", meta
    server_version = (result.stdout or "").strip().strip('"')
    if server_version:
        meta["server_version"] = server_version
    return True, "docker_ready", meta


def _web_disabled_status(settings: dict) -> tuple[bool, str]:
    cont = settings.get("continuous", {}) or {}
    autolearn = settings.get("autolearn", {}) or {}
    tools_web = (settings.get("tools", {}) or {}).get("web", {}) or {}
    allowlist = resolve_web_allowlist(settings)
    if bool(tools_web.get("enabled", False)):
        return False, "tools.web.enabled=true"
    if bool(cont.get("ingest_web", False)):
        return False, "continuous.ingest_web=true"
    if bool(autolearn.get("web_ingest", False)):
        return False, "autolearn.web_ingest=true"
    if bool(autolearn.get("url_discovery", False)):
        return False, "autolearn.url_discovery=true"
    if not resolve_web_strict(settings):
        return False, "security.web.strict=false"
    if allowlist:
        return False, "web_allowlist_not_empty"
    return True, "web_disabled"


def _external_engine_status(core: dict, *, base_dir: Path | None = None) -> tuple[bool, str, dict[str, Any], list[str]]:
    backend = str(core.get("backend", "vortex") or "vortex").strip().lower()
    engine = str(core.get("external_engine") or core.get("engine") or backend).strip().lower()
    base_url = str(core.get("external_base_url") or core.get("external_url") or "").strip()
    model_name = str(core.get("external_model") or "").strip()
    meta: dict[str, Any] = {
        "engine": engine,
        "base_url": base_url or None,
        "model": model_name or None,
        "base_url_local": _is_local_base_url(base_url),
    }
    next_steps: list[str] = []
    if base_dir is not None:
        try:
            cache_dir = resolve_cache_dir(base_dir / "data" / "models" / "hf-cache")
            cache_meta = model_cache_status(model_name or DEFAULT_MODEL_ID, cache_dir)
            meta["model_cache"] = cache_meta
        except Exception:
            meta["model_cache"] = None
    if backend not in {"external", "vllm"}:
        return True, "not_external", meta, next_steps
    if not base_url:
        return False, "external_base_url_missing", meta, next_steps
    if engine == "ollama":
        try:
            payload = _http_json(base_url.rstrip("/") + "/api/tags", timeout_s=2.0)
        except Exception as exc:
            next_steps.extend(["Start Ollama locally with `ollama serve`.", f"Pull the model with `ollama pull {model_name}`." if model_name else "Pull the required Ollama model."])
            meta["detail"] = str(exc)
            return False, "ollama_unreachable", meta, next_steps
        models = payload.get("models", []) if isinstance(payload.get("models"), list) else []
        names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
        meta["available_models"] = names
        if model_name and model_name not in names:
            next_steps.append(f"Pull the model with `ollama pull {model_name}`.")
            return False, "ollama_model_missing", meta, next_steps
        return True, "ollama_ready", meta, next_steps
    if engine in {"lmstudio", "vllm"}:
        try:
            payload = _http_json(base_url.rstrip("/") + "/v1/models", timeout_s=2.0)
        except Exception as exc:
            meta["detail"] = str(exc)
            return False, f"{engine}_unreachable", meta, next_steps
        data = payload.get("data", []) if isinstance(payload.get("data"), list) else []
        ids = [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]
        meta["available_models"] = ids
        if model_name and model_name not in ids:
            cache_meta = meta.get("model_cache") if isinstance(meta.get("model_cache"), dict) else {}
            return False, f"{engine}_model_missing", meta, next_steps
        return True, f"{engine}_ready", meta, next_steps
    return True, "external_engine_not_checked", meta, next_steps


def _wsl_ready_status(settings: dict) -> tuple[bool, str]:
    contract = settings.get("profile_contract", {}) or {}
    server_cfg = settings.get("server", {}) or {}
    strategy = str(server_cfg.get("train_strategy", "") or "").strip().lower()
    if not bool(contract.get("require_wsl_training", False)) and strategy != "wsl_subprocess_unload":
        return True, "wsl_not_required"
    try:
        from .utils.wsl import is_wsl_available
    except Exception as exc:
        return False, f"wsl_check_unavailable:{exc}"
    status = is_wsl_available(timeout_s=1.5)
    if not bool(status.ok):
        return False, status.error or "wsl_unavailable"
    wsl_workdir = str(server_cfg.get("wsl_workdir", "") or "").strip()
    if not wsl_workdir:
        return False, "wsl_workdir_missing"
    if not wsl_workdir.startswith("/mnt/"):
        return False, "wsl_workdir_not_mnt"
    return True, "wsl_ready"


def _training_ready_status(settings: dict, *, web_disabled: bool, wsl_ready: bool) -> tuple[bool, str]:
    hf_train = settings.get("hf_train", {}) or {}
    if not bool(hf_train.get("enabled", False)):
        return False, "hf_train_disabled"
    if not web_disabled:
        return False, "web_not_disabled_for_training"
    if not (hf_train.get("model_name") or (settings.get("core", {}) or {}).get("hf_model")):
        return False, "training_model_missing"
    contract = settings.get("profile_contract", {}) or {}
    if bool(contract.get("require_wsl_training", False)) and not wsl_ready:
        return False, "wsl_not_ready"
    if bool(contract.get("approved_training_sources_only", False)):
        cont = settings.get("continuous", {}) or {}
        local_sources = cont.get("local_sources", {}) or {}
        if not bool(local_sources.get("enabled", False)):
            return False, "local_sources_disabled"
        if not bool(local_sources.get("include_repo", False)):
            return False, "local_repo_disabled"
        if not bool(local_sources.get("include_local_corpus", False)):
            return False, "local_corpus_disabled"
        if not bool(local_sources.get("include_lessons", False)):
            return False, "local_lessons_disabled"
        if bool(local_sources.get("include_logs", True)):
            return False, "logs_enabled"
        if bool(local_sources.get("include_memory", True)):
            return False, "memory_enabled"
        extra_paths = hf_train.get("extra_training_paths", []) or []
        if not isinstance(extra_paths, list) or not extra_paths:
            return False, "extra_training_paths_missing"
    return True, "training_ready"


def _resolve_active_model(settings: dict, *, backend_requested: str, backend_resolved: str, external_meta: dict[str, Any], llama: dict[str, Any]) -> str | None:
    core = settings.get("core", {}) or {}
    if backend_requested == "external":
        return str(external_meta.get("model") or core.get("external_model") or "").strip() or None
    if backend_resolved == "llama_cpp":
        return str(llama.get("gguf_path") or core.get("llama_cpp_model_path") or "").strip() or None
    if backend_resolved == "hf":
        return str(core.get("hf_model") or "").strip() or None
    return str(core.get("model_name") or "").strip() or None


def _llama_cpp_ready(core: dict, *, base_dir: Path) -> dict[str, Any]:
    gguf = _resolve_path(base_dir, core.get("llama_cpp_model_path"))
    if gguf is None:
        return {"ok": False, "error": "gguf_path_missing", "gguf_path": None}
    if not gguf.exists():
        return {"ok": False, "error": "gguf_missing", "gguf_path": str(gguf)}
    if importlib.util.find_spec("llama_cpp") is None:
        return {"ok": False, "error": "llama_cpp_python_missing", "gguf_path": str(gguf), "install": 'python -m pip install -e ".[llama_cpp]"'}
    return {"ok": True, "gguf_path": str(gguf)}


def _hf_quant_status(core: dict) -> dict[str, Any]:
    requested_4 = bool(core.get("hf_load_in_4bit"))
    requested_8 = bool(core.get("hf_load_in_8bit"))
    requested = bool(requested_4 or requested_8)
    bnb = importlib.util.find_spec("bitsandbytes") is not None
    mode = None
    if requested and bnb:
        mode = "4bit" if requested_4 else "8bit"
    return {"requested": requested, "bitsandbytes_available": bool(bnb), "quant_mode": mode}


def _hf_offload_status(core: dict, *, base_dir: Path) -> dict[str, Any]:
    device_map = core.get("hf_device_map")
    max_memory = core.get("hf_max_memory")
    offload_folder_raw = core.get("hf_offload_folder")
    offload_folder = _resolve_path(base_dir, offload_folder_raw)
    enabled = bool(device_map or max_memory or offload_folder_raw)
    ok = bool(device_map) and bool(max_memory) and bool(offload_folder_raw)
    return {
        "enabled": bool(enabled),
        "ok": bool(ok),
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_folder": str(offload_folder) if offload_folder is not None else None,
    }


def prepare_model_state(settings: dict, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Offline (no downloads) validation of an inference configuration."""
    base_dir = Path(base_dir or ".").resolve()
    profile = str(settings.get("_profile") or "").strip() or "unknown"
    core = settings.get("core", {}) or {}
    contract = settings.get("profile_contract", {}) or {}
    backend_requested = str(core.get("backend", "vortex")).strip().lower()
    prefer_llama = _truthy(core.get("prefer_llama_cpp_if_available"))

    thresholds = settings.get("bench_thresholds", {}) or {}
    required_ctx = thresholds.get("required_ctx")
    try:
        required_ctx_i = int(required_ctx) if required_ctx is not None else None
    except Exception:
        required_ctx_i = None

    llama = _llama_cpp_ready(core, base_dir=base_dir)
    hf_quant = _hf_quant_status(core)
    hf_offload = _hf_offload_status(core, base_dir=base_dir)
    hf_device = str(core.get("hf_device") or "").strip().lower() or None
    hf_model = str(core.get("hf_model") or "").strip()
    hf_cache_dir = resolve_cache_dir(core.get("hf_cache_dir") or (base_dir / "data" / "models" / "hf-cache"))
    hf_cache = model_cache_status(hf_model or DEFAULT_MODEL_ID, hf_cache_dir)

    warnings: list[str] = []
    errors: list[str] = []
    next_steps: list[str] = []

    if hf_quant.get("requested") and not hf_quant.get("bitsandbytes_available"):
        warnings.append("bitsandbytes_missing_for_quant")
    if hf_offload.get("enabled") and not hf_offload.get("ok"):
        warnings.append("hf_offload_incomplete")

    hf_safe = bool(hf_quant.get("quant_mode")) or bool(hf_offload.get("ok")) or hf_device == "cpu"

    backend_resolved = backend_requested
    if backend_requested in {"hf", "transformers"}:
        backend_resolved = "hf"
        if prefer_llama and bool(llama.get("ok", False)):
            backend_resolved = "llama_cpp"
    elif backend_requested == "llama_cpp":
        backend_resolved = "llama_cpp"
        if not bool(llama.get("ok", False)):
            errors.append(str(llama.get("error") or "llama_cpp_not_ready"))

    llama_ctx = core.get("llama_cpp_ctx")
    llama_threads = core.get("llama_cpp_threads")
    try:
        llama_ctx_i = int(llama_ctx) if llama_ctx is not None else None
    except Exception:
        llama_ctx_i = None
    try:
        llama_threads_i = int(llama_threads) if llama_threads is not None else None
    except Exception:
        llama_threads_i = None

    if backend_resolved == "llama_cpp":
        if llama_ctx_i is not None and llama_ctx_i <= 0:
            errors.append("llama_cpp_ctx_invalid")
        if required_ctx_i is not None and llama_ctx_i is not None and llama_ctx_i < required_ctx_i:
            warnings.append("llama_cpp_ctx_below_required_ctx")
        if llama_threads_i is not None and llama_threads_i <= 0:
            errors.append("llama_cpp_threads_invalid")

    web_disabled, web_reason = _web_disabled_status(settings)
    wsl_ready, wsl_reason = _wsl_ready_status(settings)
    external_ready, external_reason, external_meta, external_steps = _external_engine_status(core, base_dir=base_dir)
    docker_ready, docker_reason, docker_meta = _docker_ready_status(
        settings, base_dir=base_dir
    )
    next_steps.extend(external_steps)

    ollama_ready = True
    ollama_reason = "ollama_not_required"
    if str(external_meta.get("engine") or "").lower() == "ollama":
        ollama_ready = bool(external_ready)
        ollama_reason = external_reason
    elif bool(contract.get("require_ollama", False)):
        ollama_ready = False
        ollama_reason = "ollama_required"

    training_ready, training_reason = _training_ready_status(
        settings,
        web_disabled=web_disabled,
        wsl_ready=wsl_ready,
    )

    engine_kind = (
        str(external_meta.get("engine") or backend_resolved or backend_requested).strip().lower()
        if backend_requested == "external"
        else str(backend_resolved or backend_requested).strip().lower()
    )
    engine_ready = True
    engine_reason = "engine_not_required"
    if backend_requested == "external":
        engine_ready = bool(external_ready)
        engine_reason = external_reason
    elif backend_resolved == "llama_cpp":
        engine_ready = bool(llama.get("ok", False))
        engine_reason = str(llama.get("error") or "llama_cpp_ready")
    active_model = _resolve_active_model(
        settings,
        backend_requested=backend_requested,
        backend_resolved=backend_resolved,
        external_meta=external_meta,
        llama=llama,
    )
    model_ready = bool(active_model)
    model_reason = "model_not_required"
    if backend_requested == "external":
        model_ready = bool(external_ready and active_model)
        model_reason = "model_ready" if model_ready else external_reason
    elif backend_resolved == "llama_cpp":
        model_ready = bool(llama.get("ok", False) and active_model)
        model_reason = "model_ready" if model_ready else str(llama.get("error") or "llama_cpp_not_ready")
    elif backend_resolved == "hf":
        model_ready = bool(active_model and hf_cache.get("cached", False))
        model_reason = "model_ready" if model_ready else (
            "hf_model_cache_missing" if active_model else "hf_model_missing"
        )

    offline_ready = bool(web_disabled)
    offline_reason = web_reason
    if backend_requested == "external":
        if not bool(external_meta.get("base_url_local", False)):
            offline_ready = False
            offline_reason = "external_base_url_not_local"
        elif not external_ready:
            offline_ready = False
            offline_reason = external_reason
    elif backend_resolved == "llama_cpp" and not bool(llama.get("ok", False)):
        offline_ready = False
        offline_reason = str(llama.get("error") or "llama_cpp_not_ready")
    elif backend_resolved == "hf" and not bool(hf_cache.get("cached", False)):
        offline_ready = False
        offline_reason = "hf_model_cache_missing"

    if backend_requested == "external" and not external_ready:
        errors.append(external_reason)
    if bool(contract.get("require_docker", False)) and not docker_ready:
        errors.append(f"docker_not_ready:{docker_reason}")
    if bool(contract.get("offline_required", False)) and not offline_ready:
        errors.append(f"offline_not_ready:{offline_reason}")
    if bool(contract.get("require_web_disabled", False)) and not web_disabled:
        errors.append(f"web_not_disabled:{web_reason}")
    if bool(contract.get("require_ollama", False)) and not ollama_ready:
        errors.append(f"ollama_not_ready:{ollama_reason}")
    if bool(contract.get("require_wsl_training", False)) and not wsl_ready:
        errors.append(f"wsl_not_ready:{wsl_reason}")
    if bool(contract.get("approved_training_sources_only", False)) and not training_ready:
        errors.append(f"training_not_ready:{training_reason}")
    required_engine = contract.get("require_external_engine")
    if required_engine and str(required_engine).strip().lower() != engine_kind:
        errors.append(f"engine_not_ready:expected_{str(required_engine).strip().lower()}_got_{engine_kind}")

    if not web_disabled:
        next_steps.append("Disable tools.web, continuous.ingest_web, autolearn.web_ingest and autolearn.url_discovery; keep the web allowlist empty.")
    if bool(contract.get("require_wsl_training", False)) and not wsl_ready:
        next_steps.append("Install and initialize WSL2, then set server.wsl_workdir to the repo mounted under /mnt/.")
    if bool(contract.get("require_docker", False)) and not docker_ready:
        next_steps.append("Start Docker Desktop and verify `docker info` succeeds.")
    if backend_resolved == "hf" and not bool(hf_cache.get("cached", False)):
        next_steps.append(
            "Populate the configured HF model explicitly: python -m c3rnt2.model_init "
            f"--model {active_model or DEFAULT_MODEL_ID} --cache-dir {hf_cache_dir} --download"
        )
    next_steps.append(f"Re-run: python -m vortex prepare-model --profile {profile}")

    optional_training_disabled = training_reason == "hf_train_disabled"
    degraded_reason = None
    for candidate in (
        None if docker_ready else docker_reason,
        None if offline_ready else offline_reason,
        None if engine_ready else engine_reason,
        None if model_ready else model_reason,
        None if web_disabled else web_reason,
        None if training_ready or optional_training_disabled else training_reason,
    ):
        if candidate:
            degraded_reason = candidate
            break

    state = {
        "ok": not errors,
        "ts": time.time(),
        "profile": profile,
        "cwd": str(base_dir),
        "os": os.name,
        "platform": sys.platform,
        "backend_requested": backend_requested,
        "backend_resolved": backend_resolved,
        "hf": {
            "device": hf_device,
            "quant": hf_quant,
            "offload": hf_offload,
            "cache": hf_cache,
            "safe": bool(hf_safe),
        },
        "llama_cpp": {
            "ready": bool(llama.get("ok", False)),
            "gguf_path": llama.get("gguf_path"),
            "ctx": llama_ctx_i,
            "threads": llama_threads_i,
            "required_ctx": required_ctx_i,
        },
        "external": external_meta,
        "docker": docker_meta,
        "docker_ready": bool(docker_ready),
        "docker_reason": docker_reason,
        "offline_ready": bool(offline_ready),
        "offline_reason": offline_reason,
        "engine_ready": bool(engine_ready),
        "engine_kind": engine_kind,
        "engine_base_url": external_meta.get("base_url"),
        "engine_reason": engine_reason,
        "model_ready": bool(model_ready),
        "model_reason": model_reason,
        "active_backend": backend_resolved,
        "active_model": active_model,
        "ollama_ready": bool(ollama_ready),
        "ollama_reason": ollama_reason,
        "wsl_ready": bool(wsl_ready),
        "wsl_reason": wsl_reason,
        "web_disabled": bool(web_disabled),
        "web_reason": web_reason,
        "training_ready": bool(training_ready),
        "training_reason": training_reason,
        "degraded_reason": degraded_reason,
        "quant_mode": hf_quant.get("quant_mode"),
        "offload_enabled": bool(hf_offload.get("enabled")),
        "gguf_path": llama.get("gguf_path"),
        "warnings": warnings or None,
        "errors": errors or None,
        "next_steps": next_steps or None,
    }
    return state


def write_prepared_state(state: dict[str, Any], *, base_dir: Path | None = None) -> Path:
    base_dir = Path(base_dir or ".").resolve()
    profile = str(state.get("profile") or "unknown").strip() or "unknown"
    out_path = base_dir / "data" / "models" / f"prepared_{profile}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path
