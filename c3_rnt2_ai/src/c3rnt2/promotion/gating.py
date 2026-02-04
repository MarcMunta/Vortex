from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_BENCH_PROMPT = "Explain what a context manager is in Python and give a short example."


def log_promotion_decision(base_dir: Path, payload: dict[str, Any]) -> None:
    path = base_dir / "data" / "logs" / "promotions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("ts", time.time())
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        # Never fail promotion on logging issues.
        pass


def resolve_bench_thresholds(settings: dict) -> dict[str, Any]:
    bench_cfg = settings.get("bench", {}) or {}
    bench_thresholds = settings.get("bench_thresholds", {}) or {}
    autopilot_cfg = settings.get("autopilot", {}) or {}

    min_tps = bench_cfg.get(
        "min_tokens_per_sec",
        bench_thresholds.get("min_tokens_per_sec", autopilot_cfg.get("bench_min_tokens_per_sec", 0.0)),
    )
    try:
        min_tps = float(min_tps) if min_tps is not None else 0.0
    except Exception:
        min_tps = 0.0

    required_ctx = bench_cfg.get("required_ctx")
    try:
        required_ctx = int(required_ctx) if required_ctx is not None else None
    except Exception:
        required_ctx = None

    max_vram = bench_cfg.get("max_vram_peak_mb")
    try:
        max_vram = float(max_vram) if max_vram is not None else None
    except Exception:
        max_vram = None

    max_regression = None
    if bench_cfg.get("max_regression_pct") is not None:
        try:
            max_regression = float(bench_cfg.get("max_regression_pct")) / 100.0
        except Exception:
            max_regression = None
    if max_regression is None:
        mr = bench_thresholds.get("max_regression", autopilot_cfg.get("bench_max_regression", 0.15))
        try:
            max_regression = float(mr) if mr is not None else 0.15
        except Exception:
            max_regression = 0.15

    return {
        "min_tokens_per_sec": float(min_tps),
        "required_ctx": required_ctx,
        "max_vram_peak_mb": max_vram,
        "max_regression": float(max_regression),
    }


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def bench_gate(metrics: dict[str, Any], baseline: dict[str, Any] | None, thresholds: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    cand_tps = _as_float(metrics.get("tokens_per_sec"))
    cand_vram = _as_float(metrics.get("vram_peak_mb"))
    cand_ctx = _as_int(metrics.get("ctx"))

    min_tps = _as_float(thresholds.get("min_tokens_per_sec")) or 0.0
    required_ctx = _as_int(thresholds.get("required_ctx"))
    max_vram = _as_float(thresholds.get("max_vram_peak_mb"))
    max_regression = _as_float(thresholds.get("max_regression")) or 0.0

    if min_tps > 0 and cand_tps is not None and cand_tps < min_tps:
        failures.append("below_min_tokens_per_sec")
    if required_ctx is not None and cand_ctx is not None and cand_ctx < required_ctx:
        failures.append("ctx_too_small")
    if max_vram is not None and cand_vram is not None and cand_vram > max_vram:
        failures.append("vram_peak_exceeded")

    baseline_tps = _as_float(baseline.get("tokens_per_sec")) if isinstance(baseline, dict) else None
    regression = None
    if baseline_tps is not None and baseline_tps > 0 and cand_tps is not None:
        regression = max(0.0, (baseline_tps - cand_tps) / baseline_tps)
        if max_regression > 0 and float(regression) > float(max_regression):
            failures.append("regression_exceeded")

    ok = not failures
    reason = "" if ok else ",".join(failures)
    return {
        "ok": bool(ok),
        "reason": reason,
        "failures": failures if failures else None,
        "candidate_tokens_per_sec": cand_tps,
        "baseline_tokens_per_sec": baseline_tps,
        "regression": regression,
        "min_tokens_per_sec": min_tps,
        "required_ctx": required_ctx,
        "max_vram_peak_mb": max_vram,
        "max_regression": max_regression,
    }


def compare_to_baseline(metrics: dict[str, Any], baseline: dict[str, Any] | None, thresholds: dict[str, Any]) -> tuple[bool, str]:
    verdict = bench_gate(metrics, baseline, thresholds)
    return bool(verdict.get("ok", False)), str(verdict.get("reason") or "")


def run_bench_minimal(profile: str, base_dir: Path, *, max_new_tokens: int = 64) -> dict[str, Any]:
    from ..bench import BenchArgs, run_bench
    from ..config import load_settings

    settings = load_settings(profile)
    bench_cfg = settings.get("bench", {}) or {}
    required_ctx = bench_cfg.get("required_ctx")
    try:
        ctx = int(required_ctx) if required_ctx is not None else None
    except Exception:
        ctx = None

    out_path = base_dir / "data" / "bench" / "promotion_minimal.json"
    args = BenchArgs(
        profile=str(profile),
        prompt=DEFAULT_BENCH_PROMPT,
        prompt_file=None,
        ctx=ctx,
        max_new=int(max_new_tokens),
        warmup=1,
        repeat=1,
        seed=0,
        json_out=out_path,
        jsonl_out=None,
    )
    return run_bench(settings, base_dir=base_dir, args=args)

