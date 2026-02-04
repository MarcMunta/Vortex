from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APPROVAL_FILES = ("APPROVE.txt", "APPROVE.json")


@dataclass(frozen=True)
class PromotionResult:
    ok: bool
    promoted: bool
    reason: str
    run_id: str | None = None
    adapter_path: str | None = None
    current_adapter_path: str | None = None


def quarantine_root(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "quarantine"


def promoted_root(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "promoted"


def current_pointer_path(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "current_adapter.json"


def quarantine_run_dir(base_dir: Path, run_id: str) -> Path:
    return quarantine_root(base_dir) / str(run_id)


def promoted_run_dir(base_dir: Path, run_id: str) -> Path:
    return promoted_root(base_dir) / str(run_id)


def approval_present(run_dir: Path) -> bool:
    for name in APPROVAL_FILES:
        if (run_dir / name).exists():
            return True
    return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_promotion_request(run_dir: Path, payload: dict[str, Any]) -> Path:
    out = run_dir / "PROMOTION_REQUEST.json"
    _atomic_write_json(out, payload)
    return out


def _update_current_pointer(base_dir: Path, *, run_id: str, adapter_path: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    pointer = current_pointer_path(base_dir)
    state = _load_json(pointer)
    history = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
    current = state.get("current_adapter_path")
    current_run = state.get("current_run_id")
    if current:
        history.append({"run_id": current_run, "adapter_path": current, "ts": state.get("ts")})
    state = {
        "current_run_id": str(run_id),
        "current_adapter_path": str(adapter_path),
        "ts": time.time(),
        "history": history[-50:],
    }
    if meta:
        state["meta"] = dict(meta)
    _atomic_write_json(pointer, state)
    return state


def promote_quarantine_run(
    base_dir: Path,
    *,
    run_id: str,
    require_approval: bool = True,
) -> PromotionResult:
    qdir = quarantine_run_dir(base_dir, run_id)
    adapter_src = qdir / "adapter.pt"
    manifest_path = qdir / "manifest.json"
    if not qdir.exists():
        return PromotionResult(ok=False, promoted=False, reason="quarantine_missing", run_id=run_id)
    if not adapter_src.exists():
        return PromotionResult(ok=False, promoted=False, reason="adapter_missing", run_id=run_id)

    manifest = _load_json(manifest_path)
    passed_eval = bool(manifest.get("passed_eval", False))
    if not passed_eval:
        return PromotionResult(ok=True, promoted=False, reason="passed_eval_false", run_id=run_id)

    if require_approval and not approval_present(qdir):
        return PromotionResult(ok=True, promoted=False, reason="approval_missing", run_id=run_id)

    pdir = promoted_run_dir(base_dir, run_id)
    pdir.mkdir(parents=True, exist_ok=True)
    adapter_dst = pdir / "adapter.pt"
    shutil.copy2(adapter_src, adapter_dst)
    if manifest_path.exists():
        shutil.copy2(manifest_path, pdir / "manifest.json")
    promo_req = qdir / "PROMOTION_REQUEST.json"
    if promo_req.exists():
        shutil.copy2(promo_req, pdir / "PROMOTION_REQUEST.json")

    state = _update_current_pointer(base_dir, run_id=run_id, adapter_path=adapter_dst, meta={"source": "quarantine"})
    return PromotionResult(
        ok=True,
        promoted=True,
        reason="promoted",
        run_id=str(run_id),
        adapter_path=str(adapter_dst),
        current_adapter_path=str(state.get("current_adapter_path")),
    )

