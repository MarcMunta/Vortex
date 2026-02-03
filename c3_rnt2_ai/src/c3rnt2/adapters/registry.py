from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterRegistry:
    enabled: bool
    paths: dict[str, str]
    max_loaded: int
    default: str | None = None

    @classmethod
    def from_settings(cls, settings: dict, base_dir: Path) -> "AdapterRegistry":
        cfg = settings.get("adapters", {}) or {}
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
        if not paths:
            enabled = False
        if default and default not in paths:
            # Ignore invalid default, but keep registry valid.
            default = None
        return cls(enabled=enabled, paths=paths, max_loaded=max_loaded, default=default)

    def get_path(self, name: str) -> str | None:
        return self.paths.get(str(name))

    @property
    def names(self) -> list[str]:
        return sorted(self.paths.keys())

