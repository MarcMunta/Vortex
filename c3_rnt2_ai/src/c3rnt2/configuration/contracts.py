from __future__ import annotations

import os
import warnings
from urllib.parse import urlparse


def resolve_web_allowlist(settings: dict) -> list[str]:
    security = settings.get("security", {}) or {}
    web_sec = security.get("web", {}) or {}
    allowlist_domains = web_sec.get("allowlist_domains")
    if isinstance(allowlist_domains, list):
        return [str(item) for item in allowlist_domains if item]
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    if isinstance(web_cfg.get("allow_domains"), list) and web_cfg.get("allow_domains"):
        return [str(item) for item in web_cfg.get("allow_domains") if item]
    agent_cfg = settings.get("agent", {}) or {}
    return [str(item) for item in agent_cfg.get("web_allowlist", []) if item]

def resolve_web_strict(settings: dict) -> bool:
    security = settings.get("security", {}) or {}
    web_sec = security.get("web", {}) or {}
    strict = web_sec.get("strict")
    if strict is None:
        return True
    return bool(strict)

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
    if str(os.getenv("C3RNT2_ASSUME_DOCKER_READY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        if host.endswith(".docker.internal"):
            return True
        if host and "." not in host:
            return True
    return False

def _get_nested(d: dict, keys: list[str], default: object = None) -> object:
    cur: object = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default

def _set_nested(d: dict, keys: list[str], value: object) -> None:
    cur = d
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value

def _clamp_int(settings: dict, keys: list[str], *, max_value: int, label: str) -> None:
    raw = _get_nested(settings, keys, None)
    if raw is None:
        return
    try:
        val = int(raw)
    except Exception:
        return
    if val > int(max_value):
        warnings.warn(
            f"{label} clamped to {int(max_value)} for rtx4080_16gb_safe (was {val})"
        )
        _set_nested(settings, keys, int(max_value))

def _apply_rtx4080_16gb_safe_clamps(settings: dict) -> dict:
    # Hard safety clamps for RTX 4080 16GB profiles. These are NOT usage quotas;
    # they prevent runaway VRAM/ctx/cache settings that can OOM the process.
    profile = str(settings.get("_profile") or "")
    if profile != "rtx4080_16gb_safe":
        return settings

    _clamp_int(
        settings,
        ["decode", "max_new_tokens"],
        max_value=128,
        label="decode.max_new_tokens",
    )
    _clamp_int(
        settings, ["core", "local_window"], max_value=2048, label="core.local_window"
    )
    _clamp_int(settings, ["kv", "window_size"], max_value=2048, label="kv.window_size")
    _clamp_int(
        settings,
        ["runtime", "cache_vram_budget_mb"],
        max_value=4096,
        label="runtime.cache_vram_budget_mb",
    )
    _clamp_int(
        settings,
        ["c3", "cache_vram_budget_mb"],
        max_value=4096,
        label="c3.cache_vram_budget_mb",
    )
    _clamp_int(
        settings,
        ["runtime", "prefetch_depth"],
        max_value=4,
        label="runtime.prefetch_depth",
    )
    _clamp_int(
        settings,
        ["continuous", "batch_tokens"],
        max_value=8192,
        label="continuous.batch_tokens",
    )

    runtime = dict(settings.get("runtime", {}) or {})
    if str(runtime.get("kv_quant_2bit_experimental", "")).lower() not in {
        "1",
        "true",
        "yes",
    }:
        runtime["kv_quant_2bit_experimental"] = False
    settings["runtime"] = runtime
    return settings

def _validate_profile_contract(settings: dict, errors: list[str]) -> None:
    contract = settings.get("profile_contract", {}) or {}
    if not contract:
        return

    core = settings.get("core", {}) or {}
    cont = settings.get("continuous", {}) or {}
    autolearn = settings.get("autolearn", {}) or {}
    tools_web = (settings.get("tools", {}) or {}).get("web", {}) or {}
    docker_cfg = settings.get("docker", {}) or {}
    allowlist = resolve_web_allowlist(settings)
    strict = resolve_web_strict(settings)

    backend = str(core.get("backend", "vortex")).strip().lower()
    fallback_backend = core.get("backend_fallback")
    hf_fallback = core.get("hf_fallback")
    base_url = core.get("external_base_url") or core.get("external_url")
    local_base_url_ok = _is_local_base_url(base_url)
    configured_engine = str(
        core.get("external_engine") or core.get("engine") or backend
    ).strip().lower()

    offline_required = bool(contract.get("offline_required", False))
    require_web_disabled = bool(contract.get("require_web_disabled", False))
    if offline_required or require_web_disabled:
        if bool(tools_web.get("enabled", False)):
            errors.append("profile_contract requires tools.web.enabled=false")
        if bool(cont.get("ingest_web", False)):
            errors.append("profile_contract requires continuous.ingest_web=false")
        if bool(autolearn.get("web_ingest", False)):
            errors.append("profile_contract requires autolearn.web_ingest=false")
        if bool(autolearn.get("url_discovery", False)):
            errors.append("profile_contract requires autolearn.url_discovery=false")
        if not strict:
            errors.append("profile_contract requires security.web.strict=true")
        tools_allow_domains = tools_web.get("allow_domains")
        if isinstance(tools_allow_domains, list) and any(str(item).strip() for item in tools_allow_domains):
            errors.append("profile_contract requires tools.web.allow_domains=[]")
        security_allowlist = ((settings.get("security", {}) or {}).get("web", {}) or {}).get(
            "allowlist_domains"
        )
        if isinstance(security_allowlist, list) and any(
            str(item).strip() for item in security_allowlist
        ):
            errors.append("profile_contract requires security.web.allowlist_domains=[]")
        agent_allowlist = (settings.get("agent", {}) or {}).get("web_allowlist")
        if isinstance(agent_allowlist, list) and any(str(item).strip() for item in agent_allowlist):
            errors.append("profile_contract requires agent.web_allowlist=[]")
        if allowlist:
            errors.append("profile_contract requires empty web allowlist")
    if offline_required:
        if isinstance(fallback_backend, str):
            fallback_disabled = fallback_backend.strip().lower() in {"", "none", "null"}
        else:
            fallback_disabled = fallback_backend is None
        if not fallback_disabled:
            errors.append("profile_contract requires core.backend_fallback=null")
        if backend == "external" and not local_base_url_ok:
            errors.append("profile_contract requires localhost external_base_url for external backend")

    disable_fallbacks = bool(contract.get("disable_fallbacks", False))
    if disable_fallbacks:
        if isinstance(fallback_backend, str):
            fallback_disabled = fallback_backend.strip().lower() in {"", "none", "null"}
        else:
            fallback_disabled = fallback_backend is None
        if not fallback_disabled:
            errors.append("profile_contract requires core.backend_fallback=null")
        if isinstance(hf_fallback, str):
            hf_fallback_disabled = hf_fallback.strip().lower() in {"", "none", "null"}
        else:
            hf_fallback_disabled = hf_fallback is None
        if not hf_fallback_disabled:
            errors.append("profile_contract requires core.hf_fallback=null")
        if bool(core.get("allow_implicit_hf_fallback", True)):
            errors.append("profile_contract requires core.allow_implicit_hf_fallback=false")

    require_ollama = bool(contract.get("require_ollama", False))
    required_engine = contract.get("require_external_engine")
    require_local_base_url = bool(contract.get("require_local_base_url", False))
    if require_ollama:
        if backend != "external":
            errors.append("profile_contract requires core.backend=external")
        if configured_engine != "ollama":
            errors.append("profile_contract requires core.external_engine=ollama")
    if required_engine:
        required_engine_s = str(required_engine).strip().lower()
        if backend != "external":
            errors.append("profile_contract requires core.backend=external")
        if configured_engine != required_engine_s:
            errors.append(
                f"profile_contract requires core.external_engine={required_engine_s}"
            )
    if require_ollama or require_local_base_url:
        if not local_base_url_ok:
            errors.append("profile_contract requires a localhost external_base_url")

    require_docker = bool(contract.get("require_docker", False))
    if require_docker:
        if not bool(docker_cfg.get("enabled", False)):
            errors.append("profile_contract requires docker.enabled=true")
        compose_path = docker_cfg.get("compose_path")
        if not compose_path:
            errors.append("profile_contract requires docker.compose_path")

    require_wsl_training = bool(contract.get("require_wsl_training", False))
    if require_wsl_training:
        server_cfg = settings.get("server", {}) or {}
        strategy = str(server_cfg.get("train_strategy", "") or "").strip().lower()
        if strategy != "wsl_subprocess_unload":
            errors.append("profile_contract requires server.train_strategy=wsl_subprocess_unload")
        wsl_workdir = str(server_cfg.get("wsl_workdir", "") or "").strip()
        if not wsl_workdir:
            errors.append("profile_contract requires server.wsl_workdir")
        elif not wsl_workdir.startswith("/mnt/"):
            errors.append("profile_contract requires server.wsl_workdir under /mnt/")

    approved_training_sources_only = bool(contract.get("approved_training_sources_only", False))
    hf_train_cfg = settings.get("hf_train", {}) or {}
    if approved_training_sources_only and bool(hf_train_cfg.get("enabled", False)):
        local_sources = cont.get("local_sources", {}) or {}
        if not bool(local_sources.get("enabled", False)):
            errors.append("profile_contract requires continuous.local_sources.enabled=true")
        if not bool(local_sources.get("include_repo", False)):
            errors.append("profile_contract requires local_sources.include_repo=true")
        if not bool(local_sources.get("include_local_corpus", False)):
            errors.append("profile_contract requires local_sources.include_local_corpus=true")
        if not bool(local_sources.get("include_lessons", False)):
            errors.append("profile_contract requires local_sources.include_lessons=true")
        if bool(local_sources.get("include_logs", True)):
            errors.append("profile_contract requires local_sources.include_logs=false")
        if bool(local_sources.get("include_memory", True)):
            errors.append("profile_contract requires local_sources.include_memory=false")
        extra_paths = hf_train_cfg.get("extra_training_paths", []) or []
        if not isinstance(extra_paths, list) or not extra_paths:
            errors.append("profile_contract requires hf_train.extra_training_paths")
