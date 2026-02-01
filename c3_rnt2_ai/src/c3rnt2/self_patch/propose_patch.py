from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from .utils import PatchMeta


def propose_patch(
    base_dir: Path,
    goal: str,
    context: Optional[str] = None,
    diff_text: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    patch_id = uuid.uuid4().hex[:10]
    queue_dir = base_dir / "data" / "self_patch" / "queue" / patch_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_path = queue_dir / "patch.diff"
    if diff_text is None:
        diff_text = ""
    if not dry_run:
        patch_path.write_text(diff_text, encoding="utf-8")
    else:
        patch_path.write_text(diff_text, encoding="utf-8")
    meta = PatchMeta(
        patch_id=patch_id,
        goal=goal,
        context=context,
        created_ts=time.time(),
        status="proposed",
    )
    meta_path = queue_dir / "meta.json"
    meta_path.write_text(json.dumps(meta.__dict__, ensure_ascii=True), encoding="utf-8")
    return {
        "ok": True,
        "patch_id": patch_id,
        "queue_dir": str(queue_dir),
        "diff_bytes": len(diff_text.encode("utf-8")),
    }
