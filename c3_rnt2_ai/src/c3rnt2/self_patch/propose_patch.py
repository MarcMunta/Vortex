from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from .utils import PatchMeta, generate_diff
from ..utils.locks import is_lock_held
from .policy import policy_from_settings, validate_patch
from .llm_diff import generate_diff_with_llm


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
    llm_generate_diff: bool = False,
    llm_generate_fn: Optional[Callable[[str, dict, Path, dict], str]] = None,
) -> PatchProposal:
    patch_id = uuid.uuid4().hex[:10]
    queue_root = _resolve_queue_root(repo_root, settings)
    queue_dir = queue_root / patch_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    patch_path = queue_dir / "patch.diff"
    safety_cfg = settings.get("continuous", {}).get("safety", {}) if settings else {}
    if safety_cfg.get("forbid_self_patch_during_train") and is_lock_held(repo_root, "train"):
        patch_path.write_text("", encoding="utf-8")
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
            status="blocked",
            error="train_lock_active",
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

    diff_payload = diff_text or ""
    changes = {}
    if isinstance(context, dict):
        changes = context.get("changes", {}) or {}
    if not diff_payload and changes:
        diff_payload = generate_diff(repo_root, {Path(k): v for k, v in changes.items()})
    if not diff_payload and llm_generate_diff:
        if llm_generate_fn is not None:
            diff_payload = llm_generate_fn(goal, context or {}, repo_root, settings or {})
        else:
            diff_payload = generate_diff_with_llm(goal, context or {}, repo_root, settings or {})
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
    if diff_payload:
        policy = policy_from_settings(settings)
        ok, message, _paths = validate_patch(repo_root, diff_payload, policy)
        if not ok:
            meta.status = "blocked"
            meta.error = message
        elif meta.status == "proposed":
            # Task 3: Sandbox validation
            try:
                from ..agent.sandbox_evaluator import SandboxEvaluator
                evaluator = SandboxEvaluator()
                sandbox_res = evaluator.evaluate_patch(
                    repo_root, 
                    diff_payload, 
                    test_cmd="pytest && cd vortex-chat && npm run build"
                )
                if sandbox_res.get("ok"):
                    meta.status = "approved_by_sandbox"
                else:
                    # Si falla, podemos dejarlo en 'proposed' para revisión manual o en un estado de fallo
                    # Lo dejamos en proposed para no romper el flujo manual
                    pass
            except Exception as e:
                pass

    meta_path = queue_dir / "meta.json"
    meta_path.write_text(json.dumps(meta.__dict__, ensure_ascii=True), encoding="utf-8")
    return PatchProposal(
        patch_id=patch_id,
        queue_dir=queue_dir,
        patch_path=patch_path,
        meta_path=meta_path,
        dry_run=bool(dry_run),
    )
