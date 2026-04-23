from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from .constants import DEFAULT_SETTINGS_PATH, resolve_profile
from .contracts import _apply_rtx4080_16gb_safe_clamps
from .merge import _resolve_profile
from .normalize import normalize_settings
from .types import ProfileMap, ResolvedSettings, SettingsDocument, SettingsPath, YamlMapping


def _coerce_settings_path(settings_path: SettingsPath) -> Path:
    return Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH


def _load_yaml_mapping(path: Path) -> YamlMapping:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Settings document must be a mapping: {path}")
    return data


def _coerce_imports(data: YamlMapping, path: Path) -> list[str]:
    imports = data.get("imports") or []
    if not isinstance(imports, list):
        raise ValueError(f"settings imports must be a list: {path}")
    return [str(item) for item in imports]


def _coerce_profiles(data: YamlMapping, path: Path) -> ProfileMap:
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ValueError(f"settings profiles must be a mapping: {path}")
    typed_profiles: ProfileMap = {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"settings profile must be a mapping: {path}::{name}")
        typed_profiles[str(name)] = profile
    return typed_profiles


def _resolve_import_path(path: Path, import_ref: str) -> Path:
    return (path.parent / import_ref).resolve()


def _collect_settings_sources(path: Path, stack: list[Path], ordered: list[Path]) -> None:
    resolved = path.resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in stack + [resolved])
        raise ValueError(f"settings import cycle detected: {cycle}")
    data = _load_yaml_mapping(resolved)
    for import_ref in _coerce_imports(data, resolved):
        _collect_settings_sources(
            _resolve_import_path(resolved, import_ref),
            stack + [resolved],
            ordered,
        )
    if resolved not in ordered:
        ordered.append(resolved)


def resolve_settings_sources(settings_path: SettingsPath = None) -> list[Path]:
    root = _coerce_settings_path(settings_path)
    ordered: list[Path] = []
    _collect_settings_sources(root, [], ordered)
    return ordered


def load_settings_document(settings_path: SettingsPath = None) -> SettingsDocument:
    merged: ProfileMap = {}
    import_paths: list[str] = []
    for source_path in resolve_settings_sources(settings_path):
        data = _load_yaml_mapping(source_path)
        source_imports = _coerce_imports(data, source_path)
        if source_imports:
            import_paths.extend(source_imports)
        for name, profile in _coerce_profiles(data, source_path).items():
            if name in merged:
                raise ValueError(
                    f"Duplicate profile '{name}' found while loading settings from {source_path}"
                )
            merged[name] = profile
    document: SettingsDocument = {"profiles": merged}
    if import_paths:
        document["imports"] = import_paths
    return document


def load_settings(profile: str | None = None, settings_path: SettingsPath = None) -> ResolvedSettings:
    path = _coerce_settings_path(settings_path)
    data = load_settings_document(path)
    profiles = data.get("profiles", {})
    resolved = resolve_profile(profile)
    if resolved not in profiles:
        raise KeyError(f"Profile '{resolved}' not found in {path}")
    settings = normalize_settings(_resolve_profile(profiles, resolved, []))
    settings["_profile"] = resolved
    settings = _apply_rtx4080_16gb_safe_clamps(settings)
    return settings
