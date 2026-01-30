from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def resolve_profile(profile: str | None = None) -> str:
    env_profile = os.getenv("C3RNT2_PROFILE")
    return profile or env_profile or "dev_small"


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_profile(profiles: dict[str, Any], name: str, stack: list[str]) -> dict[str, Any]:
    if name in stack:
        cycle = " -> ".join(stack + [name])
        raise ValueError(f"base_profile cycle detected: {cycle}")
    if name not in profiles:
        raise KeyError(f"Profile '{name}' not found")
    profile = profiles[name]
    base_name = profile.get("base_profile")
    if base_name:
        base_profile = _resolve_profile(profiles, base_name, stack + [name])
        override = {k: v for k, v in profile.items() if k != "base_profile"}
        return _merge_dicts(base_profile, override)
    return deepcopy(profile)




def normalize_settings(settings: dict) -> dict:
    normalized = deepcopy(settings)
    tok = normalized.get("tokenizer", {}) or {}
    if "vortex_tok_path" not in tok and tok.get("vortex_model_path"):
        tok["vortex_tok_path"] = tok.get("vortex_model_path")
    normalized["tokenizer"] = tok

    runtime = normalized.get("runtime")
    c3 = normalized.get("c3")
    if runtime is None:
        runtime = {}
    runtime = dict(runtime)
    if c3:
        runtime.setdefault("paged_lm_head", True)
        runtime.setdefault("paged_tile_out", c3.get("tile_size"))
        runtime.setdefault("paged_tile_in", c3.get("tile_in"))
        runtime.setdefault("cache_vram_budget_mb", c3.get("cache_vram_budget_mb"))
        runtime.setdefault("paged_lm_head_stream_topk", c3.get("paged_lm_head_stream_topk"))
        runtime.setdefault("prefetch_depth", c3.get("prefetch_depth"))
        runtime.setdefault("compression", c3.get("compression"))
        runtime.setdefault("pinned_memory", c3.get("pinned_memory"))
    if "paged_lm_head" not in runtime:
        runtime["paged_lm_head"] = False
    if "cache_vram_budget_mb" not in runtime:
        runtime["cache_vram_budget_mb"] = 2048
    runtime.setdefault("prefetch_depth", 2)
    runtime.setdefault("paged_lm_head_stream_topk", runtime.get("paged_lm_head_stream_topk", False) or False)
    kv = normalized.get("kv", {}) or {}
    if "kv_quant" not in runtime:
        kv_bits = kv.get("kv_quant_bits")
        if kv_bits is not None:
            if int(kv_bits) == 8:
                runtime["kv_quant"] = "int8"
            elif int(kv_bits) == 2:
                runtime["kv_quant"] = "2bit"
            elif int(kv_bits) <= 0:
                runtime["kv_quant"] = "none"
    runtime.setdefault("kv_quant", "none")
    runtime.setdefault("gpu_decompress", "none")
    normalized["runtime"] = runtime

    vx = normalized.get("vortex_model", {}) or {}
    core = normalized.get("core", {}) or {}
    core.setdefault("backend", "vortex")
    if "tf32" not in core and core.get("allow_tf32") is not None:
        core["tf32"] = core.get("allow_tf32")
    normalized["core"] = core
    lava_keys = {
        "lava_top_k",
        "lava_clusters",
        "lava_cluster_top",
        "lava_read_every",
        "lava_write_every",
        "lava_write_on_surprise",
        "lava_surprise_threshold",
        "lava_cluster_ema",
        "lava_cluster_reassign_threshold",
        "lava_ann_mode",
        "lava_shared_groups",
    }
    lava = {}
    for key in lava_keys:
        if key in vx:
            lava[key] = vx.get(key)
        elif key in core:
            lava[key] = core.get(key)
    cont = normalized.get("continuous", {}) or {}
    if cont:
        if "interval_minutes" not in cont and cont.get("run_interval_minutes") is not None:
            cont["interval_minutes"] = cont.get("run_interval_minutes")
        if "max_steps_per_tick" not in cont and cont.get("max_steps") is not None:
            cont["max_steps_per_tick"] = cont.get("max_steps")
        normalized["continuous"] = cont

    if lava:
        normalized["lava"] = lava

    return normalized



def validate_profile(settings: dict, base_dir: Path | None = None) -> None:
    missing: list[str] = []
    errors: list[str] = []
    base_dir = Path(base_dir or ".").resolve()
    tok = settings.get("tokenizer", {}) or {}
    core = settings.get("core", {}) or {}
    backend = str(core.get("backend", "vortex")).lower()
    runtime = settings.get("runtime", {}) or {}
    decode = settings.get("decode", {}) or {}
    bad = settings.get("bad", {}) or {}
    cont = settings.get("continuous", {}) or {}

    if not tok.get("vortex_tok_path"):
        missing.append("tokenizer.vortex_tok_path")
    if backend == "hf":
        if not core.get("hf_model"):
            missing.append("core.hf_model")
    else:
        for key in ("hidden_size", "layers", "heads"):
            if key not in core:
                missing.append(f"core.{key}")

    if "cache_vram_budget_mb" not in runtime:
        missing.append("runtime.cache_vram_budget_mb")
    else:
        if float(runtime.get("cache_vram_budget_mb", 0)) <= 0:
            errors.append("runtime.cache_vram_budget_mb must be > 0")
    stream_topk = runtime.get("paged_lm_head_stream_topk")
    if stream_topk is not None and stream_topk is not False:
        if int(stream_topk) <= 0:
            errors.append("runtime.paged_lm_head_stream_topk must be > 0")
    prefetch_depth = runtime.get("prefetch_depth")
    if prefetch_depth is not None and int(prefetch_depth) < 0:
        errors.append("runtime.prefetch_depth must be >= 0")

    kv_quant = str(runtime.get("kv_quant", "none")).lower()
    if kv_quant not in {"none", "int8", "2bit"}:
        errors.append("runtime.kv_quant must be one of none|int8|2bit")
    gpu_decompress = str(runtime.get("gpu_decompress", "none")).lower()
    if gpu_decompress not in {"none", "triton"}:
        errors.append("runtime.gpu_decompress must be none or triton")

    top_p = float(decode.get("top_p", bad.get("top_p", 1.0)))
    if not (0.0 < top_p <= 1.0):
        errors.append("decode.top_p must be in (0, 1]")
    top_p_min_k = int(bad.get("top_p_min_k", decode.get("top_p_min_k", 0)) or 0)
    top_p_max_k = int(bad.get("top_p_max_k", decode.get("top_p_max_k", 0)) or 0)
    if top_p_min_k and top_p_max_k and top_p_min_k > top_p_max_k:
        errors.append("top_p_min_k must be <= top_p_max_k")
    draft_cfg = decode.get("draft_model", {}) or {}
    if draft_cfg.get("enabled"):
        draft_layers = int(draft_cfg.get("draft_layers", 0))
        if draft_layers <= 0:
            errors.append("decode.draft_model.draft_layers must be > 0")
        core_layers = int(core.get("layers", 0))
        if core_layers and draft_layers > core_layers:
            errors.append("decode.draft_model.draft_layers must be <= core.layers")

    interval = cont.get("interval_minutes", cont.get("run_interval_minutes"))
    if interval is not None and float(interval) <= 0:
        errors.append("continuous.interval_minutes must be > 0")
    max_steps = cont.get("max_steps_per_tick", cont.get("max_steps"))
    if max_steps is not None and int(max_steps) <= 0:
        errors.append("continuous.max_steps_per_tick must be > 0")
    lr = cont.get("lr")
    if lr is not None and float(lr) <= 0:
        errors.append("continuous.lr must be > 0")
    batch_tokens = cont.get("batch_tokens")
    if batch_tokens is not None and int(batch_tokens) <= 0:
        errors.append("continuous.batch_tokens must be > 0")

    data_root = (base_dir / "data").resolve()

    def _check_data_path(path_value: str | Path | None, label: str) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if data_root not in path.parents and path != data_root:
            errors.append(f"{label} must be under ./data")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"{label} parent not writable: {exc}")
            return
        if not os.access(path.parent, os.W_OK):
            errors.append(f"{label} parent not writable")

    _check_data_path(cont.get("knowledge_path"), "continuous.knowledge_path")
    _check_data_path(cont.get("replay", {}).get("path"), "continuous.replay.path")
    _check_data_path(cont.get("eval", {}).get("anchors_path"), "continuous.eval.anchors_path")

    if missing or errors:
        message = []
        if missing:
            message.append("missing settings keys: " + ", ".join(missing))
        if errors:
            message.append("invalid settings: " + ", ".join(errors))
        raise ValueError("; ".join(message))


def load_settings(profile: str | None = None, settings_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    resolved = resolve_profile(profile)
    if resolved not in profiles:
        raise KeyError(f"Profile '{resolved}' not found in {path}")
    return normalize_settings(_resolve_profile(profiles, resolved, []))
