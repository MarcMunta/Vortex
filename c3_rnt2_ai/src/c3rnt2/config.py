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


def load_settings(profile: str | None = None, settings_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    resolved = resolve_profile(profile)
    if resolved not in profiles:
        raise KeyError(f"Profile '{resolved}' not found in {path}")
    return _resolve_profile(profiles, resolved, [])
