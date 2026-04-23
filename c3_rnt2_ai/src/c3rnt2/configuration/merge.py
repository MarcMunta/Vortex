from __future__ import annotations

from copy import deepcopy
from typing import Any


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

def _resolve_profile(
    profiles: dict[str, Any], name: str, stack: list[str]
) -> dict[str, Any]:
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
