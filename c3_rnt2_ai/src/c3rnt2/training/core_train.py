from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..continuous.trainer import ContinualTrainer
from ..continuous.registry import load_registry
from ..continuous.promotion import promote_quarantine_run, quarantine_run_dir, write_promotion_request, approval_present


@dataclass
class CoreTrainResult:
    ok: bool
    ok_train: bool
    ok_eval: bool
    run_id: str
    adapter_dir: Path | None
    quarantine_dir: Path | None = None
    promotion_request_path: Path | None = None
    promoted: bool = False
    approval_present: bool = False
    loss: float | None = None
    steps: int = 0
    samples: int = 0
    tokens_seen: int = 0
    tokens_per_sec: float | None = None
    vram_peak_mb: float | None = None
    eval_ok: bool | None = None
    improvement: float | None = None
    regression: float | None = None
    error: str | None = None


def _load_run_meta(base_dir: Path, run_id: str) -> dict[str, Any]:
    meta_path = base_dir / "data" / "registry" / "runs" / run_id / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_latest_registry(base_dir: Path, *, adapter_path: Path | None, run_id: str) -> None:
    reg_dir = base_dir / "data" / "registry" / "core_train"
    reg_dir.mkdir(parents=True, exist_ok=True)
    registry_path = reg_dir / "registry.json"
    payload = {
        "current_adapter": str(adapter_path) if adapter_path else None,
        "last_run_id": str(run_id),
        "ts": time.time(),
    }
    try:
        registry_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass


def train_once(settings: dict, base_dir: Path, *, reuse_dataset: bool = False, max_steps: int | None = None) -> CoreTrainResult:
    local = deepcopy(settings)
    if max_steps is not None and int(max_steps) > 0:
        cont = dict(local.get("continuous", {}) or {})
        cont["max_steps_per_tick"] = int(max_steps)
        cont["max_steps"] = int(max_steps)
        local["continuous"] = cont

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    trainer = ContinualTrainer(local, base_dir)
    tick = trainer.run_tick(ingest=not bool(reuse_dataset))

    meta = _load_run_meta(base_dir, tick.run_id)
    reason = meta.get("reason")
    ok_train = reason is None
    eval_ok = bool(getattr(tick, "promoted", False)) if ok_train else False
    ok_eval = bool(eval_ok)

    adapter_path = base_dir / "data" / "registry" / "adapters" / f"{tick.run_id}.pt"
    if not adapter_path.exists():
        adapter_path = None

    # Map metrics to unified schema.
    loss = meta.get("loss")
    try:
        loss_val = float(loss) if loss is not None else float(getattr(tick, "loss", 0.0))
    except Exception:
        loss_val = None

    steps = meta.get("steps")
    try:
        steps_val = int(steps) if steps is not None else 0
    except Exception:
        steps_val = 0

    tokens_seen = meta.get("tokens_seen")
    try:
        tokens_seen_val = int(tokens_seen) if tokens_seen is not None else 0
    except Exception:
        tokens_seen_val = 0

    tps = meta.get("tokens_per_sec")
    try:
        tokens_per_sec = float(tps) if tps is not None else None
    except Exception:
        tokens_per_sec = None

    vram_peak = meta.get("gpu_mem_mb")
    try:
        vram_peak_mb = float(vram_peak) if vram_peak is not None else None
    except Exception:
        vram_peak_mb = None

    base_loss = meta.get("base_loss")
    improvement = None
    try:
        if base_loss is not None and loss_val is not None:
            improvement = float(base_loss) - float(loss_val)
    except Exception:
        improvement = None

    samples = meta.get("samples", getattr(tick, "samples", 0))
    try:
        samples_val = int(samples) if samples is not None else 0
    except Exception:
        samples_val = 0

    quarantine_dir = None
    promotion_request_path = None
    promoted = False
    has_approval = False
    # Quarantine adapters: never promote without explicit human approval.
    if adapter_path is not None:
        quarantine_dir = quarantine_run_dir(base_dir, tick.run_id)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        q_adapter = quarantine_dir / "adapter.pt"
        try:
            q_adapter.write_bytes(adapter_path.read_bytes())
        except Exception:
            q_adapter = adapter_path
        manifest = {
            "run_id": str(tick.run_id),
            "adapter_path": str(q_adapter),
            "passed_eval": bool(eval_ok),
            "loss": loss_val,
            "base_loss": meta.get("base_loss"),
            "steps": steps_val,
            "samples": samples_val,
            "tokens_seen": tokens_seen_val,
            "tokens_per_sec": tokens_per_sec,
            "vram_peak_mb": vram_peak_mb,
            "anchor_regression": meta.get("anchor_regression"),
            "gold_regression": meta.get("gold_regression"),
            "ts": time.time(),
        }
        try:
            (quarantine_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            pass
        promotion_request_path = write_promotion_request(
            quarantine_dir,
            {
                "kind": "PROMOTION_REQUEST",
                "run_id": str(tick.run_id),
                "adapter_path": str(q_adapter),
                "manifest_path": str(quarantine_dir / "manifest.json"),
                "passed_eval": bool(eval_ok),
                "manual_approval_required": True,
                "metrics": {
                    "loss": loss_val,
                    "base_loss": meta.get("base_loss"),
                    "improvement": improvement,
                    "anchor_regression": meta.get("anchor_regression"),
                    "gold_regression": meta.get("gold_regression"),
                    "tokens_seen": tokens_seen_val,
                    "tokens_per_sec": tokens_per_sec,
                    "vram_peak_mb": vram_peak_mb,
                    "samples": samples_val,
                    "steps": steps_val,
                },
                "ts": time.time(),
            },
        )
        has_approval = approval_present(quarantine_dir)
        promote_res = promote_quarantine_run(base_dir, run_id=str(tick.run_id), require_approval=True)
        promoted = bool(promote_res.promoted)

    return CoreTrainResult(
        ok=bool(ok_train and ok_eval),
        ok_train=bool(ok_train),
        ok_eval=bool(ok_eval),
        run_id=str(tick.run_id),
        adapter_dir=adapter_path,
        quarantine_dir=quarantine_dir,
        promotion_request_path=promotion_request_path,
        promoted=bool(promoted),
        approval_present=bool(has_approval),
        loss=loss_val,
        steps=steps_val,
        samples=samples_val,
        tokens_seen=tokens_seen_val,
        tokens_per_sec=tokens_per_sec,
        vram_peak_mb=vram_peak_mb,
        eval_ok=bool(eval_ok),
        improvement=improvement,
        regression=None,
        error=str(reason) if reason else None,
    )
