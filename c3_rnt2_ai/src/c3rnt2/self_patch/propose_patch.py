from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .utils import PatchMeta, generate_diff


@dataclass
class PatchProposal:
    patch_id: str
    queue_dir: Path
    patch_path: Path
    meta_path: Path
    dry_run: bool = False


def _resolve_queue_root(base_dir: Path, settings: dict | None) -> Path:
    cfg = settings.get("self_patch", {}) if settings else {}
    queue_dir = cfg.get("queue_dir")
    if queue_dir:
        queue_path = Path(queue_dir)
        if not queue_path.is_absolute():
            queue_path = base_dir / queue_path
        return queue_path
    return base_dir / "data" / "self_patch" / "queue"


def propose_patch(
    goal: str,
    context: Optional[dict],
    repo_root: Path,
    *,
    settings: dict | None = None,
    dry_run: bool = False,
    diff_text: Optional[str] = None,
) -> PatchProposal:
    patch_id = uuid.uuid4().hex[:10]
    queue_root = _resolve_queue_root(repo_root, settings)
    queue_dir = queue_root / patch_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_path = queue_dir / "patch.diff"

    diff_payload = diff_text or ""
    changes = {}
    if isinstance(context, dict):
        changes = context.get("changes", {}) or {}
    if not diff_payload and changes:
        diff_payload = generate_diff(repo_root, {Path(k): v for k, v in changes.items()})
    patch_path.write_text(diff_payload, encoding="utf-8")

    context_payload: str | None = None
    if context is not None:
        if isinstance(context, str):
            context_payload = context
        else:
            context_payload = json.dumps(context, ensure_ascii=True)
    meta = PatchMeta(
        patch_id=patch_id,
        goal=goal,
        context=context_payload,
        created_ts=time.time(),
        status="proposed",
    )
    meta_path = queue_dir / "meta.json"
    meta_path.write_text(json.dumps(meta.__dict__, ensure_ascii=True), encoding="utf-8")
    return PatchProposal(
        patch_id=patch_id,
        queue_dir=queue_dir,
        patch_path=patch_path,
        meta_path=meta_path,
        dry_run=bool(dry_run),
    )
