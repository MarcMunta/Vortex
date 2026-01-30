from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class RegistryState:
    current_run_id: str | None
    history: list[str]


def _registry_root(base_dir: Path) -> Path:
    return base_dir / "data" / "registry"


def _current_path(base_dir: Path) -> Path:
    return _registry_root(base_dir) / "current.json"


def init_registry(base_dir: Path) -> None:
    root = _registry_root(base_dir)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "adapters").mkdir(parents=True, exist_ok=True)
    current = _current_path(base_dir)
    if not current.exists():
        current.write_text(json.dumps({"current_run_id": None, "history": []}), encoding="utf-8")


def load_registry(base_dir: Path) -> RegistryState:
    init_registry(base_dir)
    payload = json.loads(_current_path(base_dir).read_text(encoding="utf-8"))
    return RegistryState(current_run_id=payload.get("current_run_id"), history=list(payload.get("history", [])))


def save_registry(base_dir: Path, state: RegistryState) -> None:
    payload = {"current_run_id": state.current_run_id, "history": state.history}
    _current_path(base_dir).write_text(json.dumps(payload), encoding="utf-8")


def begin_run(base_dir: Path) -> tuple[str, Path]:
    init_registry(base_dir)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_path = _registry_root(base_dir) / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_id, run_path


def finalize_run(base_dir: Path, run_id: str, promote: bool, meta: Dict[str, Any] | None = None) -> None:
    state = load_registry(base_dir)
    if promote:
        if state.current_run_id:
            state.history.append(state.current_run_id)
        state.current_run_id = run_id
    save_registry(base_dir, state)
    if meta:
        run_path = _registry_root(base_dir) / "runs" / run_id
        (run_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def rollback(base_dir: Path) -> str | None:
    state = load_registry(base_dir)
    if not state.history:
        return None
    state.current_run_id = state.history.pop()
    save_registry(base_dir, state)
    return state.current_run_id


def _bootstrap_path(base_dir: Path) -> Path:
    return _registry_root(base_dir) / "bootstrap.json"


def load_bootstrap(base_dir: Path) -> Dict[str, Any] | None:
    path = _bootstrap_path(base_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def mark_bootstrapped(base_dir: Path, meta: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {"bootstrapped": True, "ts": time.time()}
    if meta:
        payload.update(meta)
    path = _bootstrap_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def is_bootstrapped(base_dir: Path, settings: Dict[str, Any] | None = None) -> bool:
    meta = load_bootstrap(base_dir)
    if meta and meta.get("bootstrapped"):
        return True
    if settings:
        core = settings.get("core", {}) or {}
        ckpt = core.get("checkpoint_path")
        if ckpt and Path(ckpt).exists():
            return True
    return False

