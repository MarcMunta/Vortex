from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .model_loader import load_inference_model


@dataclass(frozen=True)
class BenchArgs:
    profile: str
    prompt: str
    prompt_file: str | None
    ctx: int | None
    max_new: int
    warmup: int
    repeat: int
    seed: int
    json_out: Path
    jsonl_out: Path | None = None


def _rss_mb() -> float | None:
    try:
        proc = psutil.Process()
        return float(proc.memory_info().rss) / 1e6
    except Exception:
        return None


def _pct_ms(values_s: list[float], pct: float) -> float | None:
    if not values_s:
        return None
    data = sorted(values_s)
    if len(data) == 1:
        return float(data[0] * 1000.0)
    k = (len(data) - 1) * (pct / 100.0)
    f = int(k)
    c = min(len(data) - 1, f + 1)
    if f == c:
        return float(data[f] * 1000.0)
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return float((d0 + d1) * 1000.0)


def _prompt_for_ctx(core: Any, ctx: int) -> tuple[str, int] | None:
    # Only supported for CoreTransformer-like models with encode_prompt/decode_ids.
    try:
        seed = "def f(x):\n    return x\n"
        ids, _ = core.encode_prompt(seed)
        if not ids:
            return None
        reps = (int(ctx) + len(ids) - 1) // len(ids)
        prompt_ids = (ids * max(1, reps))[: int(ctx)]
        prompt = core.decode_ids(prompt_ids, total_len=int(ctx))
        final_ids, _ = core.encode_prompt(prompt)
        return prompt, int(len(final_ids))
    except Exception:
        return None


def _ctx_len(model: Any, prompt: str) -> int | None:
    try:
        ids, _total = model.encode_prompt(prompt)
        return int(len(ids))
    except Exception:
        return None


def _paging_stats(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "paging_enabled": None,
        "lm_head_is_paged": None,
        "cache_hit_rate": None,
        "bytes_prefetched": None,
        "page_faults": None,
        "bytes_compressed_read": None,
        "reason": None,
        "raw": None,
    }
    lm_head = getattr(model, "lm_head", None)
    runtime_cfg = getattr(model, "runtime_cfg", {}) or {}
    out["paging_enabled"] = bool(runtime_cfg.get("paged_lm_head", False))
    if lm_head is None:
        out["reason"] = "no_lm_head"
        return out
    try:
        stats = lm_head.stats() if hasattr(lm_head, "stats") else None
    except Exception as exc:
        out["reason"] = f"stats_failed:{exc.__class__.__name__}"
        return out
    if not isinstance(stats, dict):
        out["reason"] = "stats_unavailable"
        return out
    out["raw"] = dict(stats)
    try:
        from .nn.paged_linear import PagedLinear  # local import

        out["lm_head_is_paged"] = isinstance(lm_head, PagedLinear)
    except Exception:
        out["lm_head_is_paged"] = None
    try:
        out["page_faults"] = float(stats.get("page_faults")) if stats.get("page_faults") is not None else None
        out["bytes_prefetched"] = float(stats.get("bytes_h2d")) if stats.get("bytes_h2d") is not None else None
        out["bytes_compressed_read"] = (
            float(stats.get("bytes_compressed_read")) if stats.get("bytes_compressed_read") is not None else None
        )
    except Exception:
        pass
    try:
        cache = getattr(lm_head, "cache", None)
        cache_stats = cache.stats() if cache is not None and hasattr(cache, "stats") else None
        if isinstance(cache_stats, dict):
            out["cache_hit_rate"] = float(cache_stats.get("hit_rate")) if cache_stats.get("hit_rate") is not None else None
    except Exception:
        pass
    return out


def run_bench(settings: dict, base_dir: Path, args: BenchArgs) -> dict[str, Any]:
    if args.seed is not None:
        random.seed(int(args.seed))
        if torch is not None:
            try:
                torch.manual_seed(int(args.seed))
            except Exception:
                pass

    core_cfg = settings.get("core", {}) or {}
    backend = str(core_cfg.get("backend", "vortex")).lower()
    model = load_inference_model(settings)

    prompt_text = str(args.prompt or "")
    ctx_len_prompt = _ctx_len(model, prompt_text)
    if args.ctx and args.ctx > 0:
        candidate = _prompt_for_ctx(model, int(args.ctx))
        if candidate is not None:
            prompt_text, ctx_len_prompt = candidate

    if ctx_len_prompt is None:
        ctx_len_prompt = None
    ctx_len_total = int(ctx_len_prompt + args.max_new) if isinstance(ctx_len_prompt, int) else None

    # Warmup and steady-state runs.
    warmup_tokens = max(1, min(int(args.max_new), 16))
    warmup_s: list[float] = []
    steady_s: list[float] = []

    def _generate(tokens: int) -> None:
        if hasattr(model, "generate"):
            _ = model.generate(prompt_text, max_new_tokens=int(tokens))
            return
        raise RuntimeError("model has no generate()")

    # Warmup (not counted in steady).
    for _ in range(max(0, int(args.warmup))):
        start = time.perf_counter()
        _generate(warmup_tokens)
        warmup_s.append(max(1e-9, time.perf_counter() - start))

    # Reset peak stats for steady-state measurement.
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    paging_before = _paging_stats(model)
    for _ in range(max(1, int(args.repeat))):
        start = time.perf_counter()
        _generate(int(args.max_new))
        steady_s.append(max(1e-9, time.perf_counter() - start))
    paging_after = _paging_stats(model)

    steady_total_s = float(sum(steady_s))
    steady_tokens = int(max(1, int(args.repeat)) * int(args.max_new))
    tokens_per_sec_steady = float(steady_tokens) / max(1e-9, steady_total_s)

    warmup_total_s = float(sum(warmup_s))
    warmup_tokens_total = int(max(0, int(args.warmup)) * warmup_tokens)
    tokens_per_sec_warmup = (float(warmup_tokens_total) / max(1e-9, warmup_total_s)) if warmup_tokens_total else None

    vram_peak_allocated_mb = None
    vram_peak_reserved_mb = None
    if torch is not None and torch.cuda.is_available():
        try:
            vram_peak_allocated_mb = float(torch.cuda.max_memory_allocated() / 1e6)
        except Exception:
            vram_peak_allocated_mb = None
        try:
            vram_peak_reserved_mb = float(torch.cuda.max_memory_reserved() / 1e6)
        except Exception:
            vram_peak_reserved_mb = None

    rss_mb = _rss_mb()

    report: dict[str, Any] = {
        "ok": True,
        "ts": time.time(),
        "profile": str(args.profile),
        "backend": backend,
        "seed": int(args.seed),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "ctx": int(args.ctx) if args.ctx is not None else None,
        "ctx_target": int(args.ctx) if args.ctx is not None else None,
        "ctx_len_prompt": ctx_len_prompt,
        "ctx_len_total": ctx_len_total,
        "max_new": int(args.max_new),
        "max_new_tokens": int(args.max_new),
        "tokens_per_sec_warmup": round(float(tokens_per_sec_warmup), 6) if tokens_per_sec_warmup is not None else None,
        "tokens_per_sec_steady": round(float(tokens_per_sec_steady), 6),
        "tokens_per_sec": round(float(tokens_per_sec_steady), 6),
        "latency_ms_total": round(float(steady_total_s) * 1000.0, 3),
        "latency_ms_per_token": round(float(steady_total_s) * 1000.0 / max(1, steady_tokens), 6),
        "latency_p50_ms": round(float(_pct_ms(steady_s, 50.0) or 0.0), 3) if steady_s else None,
        "latency_p95_ms": round(float(_pct_ms(steady_s, 95.0) or 0.0), 3) if steady_s else None,
        "vram_peak_mb": round(float(vram_peak_allocated_mb), 3) if vram_peak_allocated_mb is not None else None,
        "vram_peak_mb_allocated": round(float(vram_peak_allocated_mb), 3) if vram_peak_allocated_mb is not None else None,
        "vram_peak_mb_reserved": round(float(vram_peak_reserved_mb), 3) if vram_peak_reserved_mb is not None else None,
        "ram_rss_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
        # Paging counters (required keys, null if not applicable).
        "cache_hit_rate": paging_after.get("cache_hit_rate"),
        "bytes_prefetched": paging_after.get("bytes_prefetched"),
        "page_faults": paging_after.get("page_faults"),
        "page_faults_reason": paging_after.get("reason"),
        "paging": {
            "before": paging_before,
            "after": paging_after,
        },
    }

    # Write JSON output(s)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    latest_path = base_dir / "data" / "bench" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=True) + "\n")

    return report
