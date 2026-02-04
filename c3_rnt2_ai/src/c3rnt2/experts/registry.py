from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolve_dir(base_dir: Path, value: str | Path | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _discover_hf_train_adapters(base_dir: Path, settings: dict) -> dict[str, str]:
    hf_train = settings.get("hf_train", {}) or {}
    reg_dir = _resolve_dir(base_dir, hf_train.get("registry_dir"), base_dir / "data" / "registry" / "hf_train")
    if not reg_dir.exists():
        return {}
    discovered: dict[str, str] = {}
    try:
        for child in sorted(reg_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            adapter_dir = child / "adapter"
            if not adapter_dir.is_dir():
                continue
            name = child.name
            if not name.startswith("expert_"):
                name = f"expert_{name}"
            # Do not overwrite explicit user-provided entries.
            discovered.setdefault(name, str(adapter_dir))
    except Exception:
        return {}
    return discovered


def _discover_promoted_hf_experts(base_dir: Path, settings: dict) -> dict[str, str]:
    cfg = settings.get("experts", {}) or {}
    reg_dir = _resolve_dir(base_dir, cfg.get("registry_dir"), base_dir / "data" / "experts_hf" / "registry")
    root = reg_dir / "experts"
    if not root.exists():
        return {}
    discovered: dict[str, str] = {}
    try:
        for domain_dir in sorted(root.iterdir(), key=lambda p: p.name):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for version_dir in sorted(domain_dir.iterdir(), key=lambda p: p.name):
                if not version_dir.is_dir():
                    continue
                adapter_dir = version_dir / "adapter"
                if not adapter_dir.is_dir():
                    continue
                name = f"expert_{domain}_{version_dir.name}"
                discovered.setdefault(name, str(adapter_dir))
    except Exception:
        return {}
    return discovered


@dataclass(frozen=True)
class ExpertRegistry:
    enabled: bool
    paths: dict[str, str]
    max_loaded: int
    default: str | None = None

    @classmethod
    def from_settings(cls, settings: dict, base_dir: Path) -> "ExpertRegistry":
        cfg = settings.get("experts", {}) or {}
        enabled = bool(cfg.get("enabled", False))
        max_loaded = int(cfg.get("max_loaded", 0) or 0)
        if max_loaded < 0:
            max_loaded = 0
        default = cfg.get("default")
        default = str(default) if default else None

        raw_paths = cfg.get("paths", {}) or {}
        paths: dict[str, str] = {}
        for name, raw_path in raw_paths.items():
            if not name or raw_path is None:
                continue
            p = Path(str(raw_path))
            if not p.is_absolute():
                p = base_dir / p
            paths[str(name)] = str(p)

        if enabled and not paths:
            # Optional auto-discovery: promoted HF experts registry (fail-closed for 120B-like).
            discovered = _discover_promoted_hf_experts(base_dir, settings)
            if discovered:
                paths.update(discovered)
            profile = str(settings.get("_profile") or "")
            discover_hf_train = cfg.get("discover_hf_train")
            if discover_hf_train is None:
                discover_hf_train = profile != "rtx4080_16gb_120b_like"
            if not paths and bool(discover_hf_train):
                # Legacy auto-discovery: treat HF training runs as experts.
                # Keeps workflows viable without editing settings.yaml on each train.
                discovered = _discover_hf_train_adapters(base_dir, settings)
                if discovered:
                    paths.update(discovered)

        if not paths:
            enabled = False
        if default and default not in paths:
            default = None
        return cls(enabled=enabled, paths=paths, max_loaded=max_loaded, default=default)

    def get_path(self, name: str) -> str | None:
        return self.paths.get(str(name))

    @property
    def names(self) -> list[str]:
        return sorted(self.paths.keys())
