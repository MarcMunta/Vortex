from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..promotion.gating import bench_gate, log_promotion_decision, resolve_bench_thresholds


APPROVAL_FILES = ("APPROVE.txt", "APPROVE.json")
HF_EXPERT_APPROVAL_FILE = "APPROVE_PROMOTION"


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


def _build_core_model_with_adapter(settings: dict, *, adapter_path: Path | None) -> object:
    from copy import deepcopy

    from ..model.core_transformer import CoreTransformer
    from .lora import LoRAConfig, inject_lora, load_lora_state, resolve_target_modules

    model = CoreTransformer.from_settings(deepcopy(settings))
    if adapter_path is None:
        return model
    adapter_cfg = (settings.get("continuous", {}) or {}).get("adapters", {}) or {}
    lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", adapter_cfg.get("adapter_rank", 4) or 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
    strict = bool(adapter_cfg.get("strict_target_modules", False))
    targets = resolve_target_modules(adapter_cfg, strict=strict)
    inject_lora(model, lora_cfg, target_modules=targets)
    load_lora_state(model, adapter_path)
    try:
        model.adapter_path = str(adapter_path)
    except Exception:
        pass
    return model


def _bench_short(settings: dict, *, adapter_path: Path | None, max_new_tokens: int = 64) -> dict[str, Any]:
    prompt = "Explain what a context manager is in Python and give a short example."
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    model = _build_core_model_with_adapter(settings, adapter_path=adapter_path)
    start = time.perf_counter()
    _ = model.generate(prompt, max_new_tokens=int(max_new_tokens)) if hasattr(model, "generate") else ""
    elapsed = max(1e-6, time.perf_counter() - start)
    vram_peak = None
    if torch is not None and torch.cuda.is_available():
        try:
            vram_peak = float(torch.cuda.max_memory_allocated() / (1024**2))
        except Exception:
            vram_peak = None
    tokens_per_sec = float(max_new_tokens) / elapsed
    return {
        "ok": True,
        "tokens_per_sec": float(tokens_per_sec),
        "vram_peak_mb": vram_peak,
        "elapsed_s": float(elapsed),
        "max_new_tokens": int(max_new_tokens),
    }


def promote_quarantine_run(
    base_dir: Path,
    *,
    run_id: str,
    require_approval: bool = True,
    settings: dict | None = None,
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

    if settings is not None:
        thresholds = resolve_bench_thresholds(settings)
        baseline_state = _load_json(current_pointer_path(base_dir))
        baseline_path = baseline_state.get("current_adapter_path")
        baseline_adapter = Path(str(baseline_path)) if baseline_path else None
        if baseline_adapter is not None and not baseline_adapter.is_absolute():
            baseline_adapter = base_dir / baseline_adapter
        if baseline_adapter is not None and not baseline_adapter.exists():
            baseline_adapter = None

        baseline = _bench_short(settings, adapter_path=baseline_adapter, max_new_tokens=64) if baseline_adapter else None
        candidate = _bench_short(settings, adapter_path=adapter_src, max_new_tokens=64)
        verdict = bench_gate(candidate, baseline, thresholds)
        bench_ok = bool(verdict.get("ok", False))
        log_promotion_decision(
            base_dir,
            {
                "kind": "promotion_gate",
                "backend": "core_quarantine",
                "run_id": str(run_id),
                "eval_ok": bool(passed_eval),
                "bench_ok": bool(bench_ok),
                "reason": verdict.get("reason") if not bench_ok else "ok",
                "thresholds": thresholds,
                "baseline_adapter_path": str(baseline_adapter) if baseline_adapter else None,
                "candidate_adapter_path": str(adapter_src),
                "baseline": baseline,
                "candidate": candidate,
                "bench_gate": verdict,
            },
        )
        if not bench_ok:
            reason = "bench_failed"
            if verdict.get("reason"):
                reason = reason + ":" + str(verdict.get("reason"))
            return PromotionResult(ok=True, promoted=False, reason=reason, run_id=run_id)

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


def _load_profile_bench_baseline_tps(base_dir: Path, *, profile: str, backend: str) -> float | None:
    path = base_dir / "data" / "bench" / "baseline.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    prof = payload.get(str(profile))
    if not isinstance(prof, dict):
        return None
    entry = prof.get(str(backend))
    if not isinstance(entry, dict):
        return None
    val = entry.get("tokens_per_sec")
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def promote_hf_expert(
    quarantine_dir: str | Path,
    registry_dir: str | Path,
    settings: dict,
    *,
    require_manual_approval: bool = True,
) -> dict[str, Any]:
    """Promote a HF expert adapter directory from quarantine into a registry folder.

    This is intentionally fail-closed for the 120B-like profile: it requires a manual
    approval file (data/APPROVE_PROMOTION) and enforces bench_thresholds gates.
    """
    base_dir = Path(".").resolve()
    qdir = Path(str(quarantine_dir))
    if not qdir.is_absolute():
        qdir = base_dir / qdir
    run_dir = qdir.parent if qdir.name.lower() == "adapter" else qdir
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    profile = str(settings.get("_profile") or "unknown")

    domain = str(manifest.get("domain") or run_dir.parent.name or "unknown").strip() or "unknown"
    run_id = str(manifest.get("run_id") or run_dir.name or "").strip() or run_dir.name
    adapter_src_raw = manifest.get("adapter_dir") or str(run_dir / "adapter")
    adapter_src = Path(str(adapter_src_raw))
    if not adapter_src.is_absolute():
        adapter_src = base_dir / adapter_src

    if not run_dir.exists():
        return {"ok": False, "promoted": False, "reason": "quarantine_missing", "domain": domain, "run_id": run_id}
    if not adapter_src.exists():
        return {"ok": False, "promoted": False, "reason": "adapter_missing", "domain": domain, "run_id": run_id, "adapter_dir": str(adapter_src)}

    passed_eval = bool(manifest.get("passed_eval", False))
    if not passed_eval:
        log_promotion_decision(base_dir, {"kind": "promotion_gate", "backend": "hf_experts", "profile": profile, "domain": domain, "run_id": run_id, "promote_ok": False, "reason": "passed_eval_false"})
        return {"ok": True, "promoted": False, "reason": "passed_eval_false", "domain": domain, "run_id": run_id}

    if require_manual_approval and profile == "rtx4080_16gb_120b_like":
        approval = base_dir / "data" / HF_EXPERT_APPROVAL_FILE
        if not approval.exists():
            log_promotion_decision(base_dir, {"kind": "promotion_gate", "backend": "hf_experts", "profile": profile, "domain": domain, "run_id": run_id, "promote_ok": False, "reason": "approval_missing", "approval_file": str(approval)})
            return {"ok": True, "promoted": False, "reason": "approval_missing", "domain": domain, "run_id": run_id, "approval_file": str(approval)}

    thresholds = resolve_bench_thresholds(settings)
    bench = manifest.get("bench") if isinstance(manifest.get("bench"), dict) else {}
    candidate = {
        "tokens_per_sec": bench.get("tokens_per_sec"),
        "vram_peak_mb": bench.get("vram_peak_mb"),
        "ctx": bench.get("ctx"),
    }
    baseline_tps = _load_profile_bench_baseline_tps(base_dir, profile=profile, backend="hf")
    allow_no_baseline = bool((settings.get("security", {}) or {}).get("allow_promotion_without_baseline", False))
    if profile == "rtx4080_16gb_120b_like" and baseline_tps is None and not allow_no_baseline:
        log_promotion_decision(
            base_dir,
            {
                "kind": "promotion_gate",
                "backend": "hf_experts",
                "profile": profile,
                "domain": domain,
                "run_id": run_id,
                "promote_ok": False,
                "reason": "baseline_missing",
                "baseline_path": str(base_dir / "data" / "bench" / "baseline.json"),
                "thresholds": thresholds,
                "candidate": candidate,
                "manifest": str(manifest_path),
            },
        )
        return {"ok": True, "promoted": False, "reason": "baseline_missing", "domain": domain, "run_id": run_id}
    baseline = {"tokens_per_sec": baseline_tps} if baseline_tps is not None else None
    verdict = bench_gate(candidate, baseline, thresholds)
    bench_ok = bool(verdict.get("ok", False))
    if not bench_ok:
        reason = verdict.get("reason") or "bench_failed"
        log_promotion_decision(
            base_dir,
            {
                "kind": "promotion_gate",
                "backend": "hf_experts",
                "profile": profile,
                "domain": domain,
                "run_id": run_id,
                "promote_ok": False,
                "reason": reason,
                "thresholds": thresholds,
                "baseline": baseline,
                "candidate": candidate,
                "bench_gate": verdict,
                "manifest": str(manifest_path),
            },
        )
        return {"ok": True, "promoted": False, "reason": f"bench_failed:{reason}", "domain": domain, "run_id": run_id, "bench_gate": verdict}

    reg_dir = Path(str(registry_dir))
    if not reg_dir.is_absolute():
        reg_dir = base_dir / reg_dir
    version_dir = reg_dir / "experts" / domain / run_id
    adapter_dst = version_dir / "adapter"
    if adapter_dst.exists():
        return {"ok": False, "promoted": False, "reason": "already_promoted", "domain": domain, "run_id": run_id, "adapter_dir": str(adapter_dst)}
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_src, adapter_dst)
    if manifest_path.exists():
        shutil.copy2(manifest_path, version_dir / "manifest.json")

    log_promotion_decision(
        base_dir,
        {
            "kind": "promotion_gate",
            "backend": "hf_experts",
            "profile": profile,
            "domain": domain,
            "run_id": run_id,
            "promote_ok": True,
            "reason": "promoted",
            "thresholds": thresholds,
            "baseline": baseline,
            "candidate": candidate,
            "bench_gate": verdict,
            "src": str(adapter_src),
            "dst": str(adapter_dst),
        },
    )
    return {"ok": True, "promoted": True, "reason": "promoted", "domain": domain, "run_id": run_id, "adapter_dir": str(adapter_dst), "version_dir": str(version_dir)}
