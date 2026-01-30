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
    if runtime is None and c3:
        runtime = {
            "paged_lm_head": True,
            "paged_tile_out": c3.get("tile_size"),
            "paged_tile_in": c3.get("tile_in"),
            "cache_vram_budget_mb": c3.get("cache_vram_budget_mb"),
            "paged_lm_head_stream_topk": c3.get("paged_lm_head_stream_topk"),
            "prefetch_depth": c3.get("prefetch_depth"),
            "compression": c3.get("compression"),
            "pinned_memory": c3.get("pinned_memory"),
        }
        normalized["runtime"] = {k: v for k, v in runtime.items() if v is not None}
    elif runtime is not None:
        runtime = dict(runtime)
        if c3:
            runtime.setdefault("cache_vram_budget_mb", c3.get("cache_vram_budget_mb"))
            runtime.setdefault("paged_lm_head_stream_topk", c3.get("paged_lm_head_stream_topk"))
        runtime.setdefault("cache_vram_budget_mb", 2048)
        normalized["runtime"] = runtime

    vx = normalized.get("vortex_model", {}) or {}
    core = normalized.get("core", {}) or {}
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
    if lava:
        normalized["lava"] = lava

    return normalized

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
