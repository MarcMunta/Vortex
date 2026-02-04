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
    mock: bool = False


def _resolve_stream_topk(settings: dict) -> int | bool:
    runtime = settings.get("runtime", {}) or {}
    raw = runtime.get("paged_lm_head_stream_topk")
    if raw is None or raw is False:
        return False
    try:
        val = int(raw)
    except Exception:
        return True
    return int(val) if val > 0 else False


def _normalize_weights(scores: list[float]) -> list[float]:
    vals = []
    for item in scores:
        try:
            vals.append(max(0.0, float(item)))
        except Exception:
            vals.append(0.0)
    total = float(sum(vals))
    if total <= 1e-12:
        return [1.0 / float(len(scores)) for _ in scores]
    return [float(v) / total for v in vals]


def _resolve_mix_mode(settings: dict) -> str:
    for section in ("experts", "adapters"):
        cfg = settings.get(section, {}) or {}
        router_cfg = cfg.get("router", {}) or {}
        mode = router_cfg.get("mix_mode")
        if mode:
            return str(mode).strip().lower()
    return "single"


def _maybe_prepare_hf_adapters(model: Any, settings: dict, base_dir: Path, prompt: str) -> tuple[list[str], float | None, str | None]:
    """Best-effort: load+activate adapters for HF bench, returning (selected_names, load_ms, active_adapter_name)."""
    if not bool(getattr(model, "is_hf", False)):
        return [], None, None

    registry = None
    router = None
    for section in ("experts", "adapters"):
        try:
            if section == "experts":
                from .experts.registry import ExpertRegistry as _Registry  # type: ignore
                from .experts.router import ExpertRouter as _Router  # type: ignore
            else:
                from .adapters.registry import AdapterRegistry as _Registry  # type: ignore
                from .adapters.router import AdapterRouter as _Router  # type: ignore
            registry = _Registry.from_settings(settings, base_dir=base_dir)
            if bool(getattr(registry, "enabled", False)):
                router = _Router.from_settings(settings)
                break
        except Exception:
            registry = None
            router = None
            continue
    if registry is None or router is None:
        return [], None, None

    decision = router.select(prompt or "", registry.names, top_k=None)
    selected: list[str] = []
    if decision.selected_adapters:
        selected = [str(x).strip() for x in decision.selected_adapters if str(x).strip()]
    elif decision.selected_adapter:
        selected = [str(decision.selected_adapter).strip()]
    selected = [name for name in selected if name]
    if not selected:
        return [], None, None

    if not hasattr(model, "add_adapter"):
        return [], None, getattr(model, "active_adapter_name", None)

    mix_mode = _resolve_mix_mode(settings)
    start = time.perf_counter()
    loaded: list[str] = []
    try:
        try:
            setattr(model, "adapter_max_loaded", int(registry.max_loaded))
        except Exception:
            pass

        for name in selected:
            path = registry.get_path(name)
            if not path:
                continue
            if not Path(path).exists():
                continue
            try:
                model.add_adapter(name, path)
                loaded.append(name)
            except Exception:
                continue

        if not loaded:
            return [], round((time.perf_counter() - start) * 1000.0, 3), getattr(model, "active_adapter_name", None)

        # Activate selection (single or weighted mix).
        if mix_mode == "weighted" and len(loaded) > 1 and hasattr(model, "set_weighted_adapters"):
            scores = decision.scores if isinstance(decision.scores, list) and len(decision.scores) == len(loaded) else None
            weights = _normalize_weights(scores) if scores else [1.0 / float(len(loaded)) for _ in loaded]
            adapter_weights = {name: float(weight) for name, weight in zip(loaded, weights)}
            try:
                _ = bool(model.set_weighted_adapters(adapter_weights))
            except Exception:
                pass
        else:
            if hasattr(model, "set_adapter"):
                try:
                    model.set_adapter(loaded[0])
                except Exception:
                    pass
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return loaded, float(elapsed_ms), getattr(model, "active_adapter_name", None)


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
    stream_topk = _resolve_stream_topk(settings)

    prompt_text = str(args.prompt or "")
    adapter_load_ms = None
    active_adapters: list[str] = []
    adapter_active = None

    if bool(getattr(args, "mock", False)):
        ctx_len_prompt = len(prompt_text.split()) if prompt_text else 0
        ctx_len_total = int(ctx_len_prompt + args.max_new)
        tokens_per_sec_steady = 12.0
        steady_tokens = int(max(1, int(args.repeat)) * int(args.max_new))
        steady_total_s = float(steady_tokens) / max(1e-9, float(tokens_per_sec_steady))
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
            "ctx_len_prompt": int(ctx_len_prompt),
            "ctx_len_total": int(ctx_len_total),
            "max_new": int(args.max_new),
            "max_new_tokens": int(args.max_new),
            "tokens_per_sec_warmup": None,
            "tokens_per_sec_steady": round(float(tokens_per_sec_steady), 6),
            "tokens_per_sec": round(float(tokens_per_sec_steady), 6),
            "prefill_tokens_per_sec": round(float(tokens_per_sec_steady), 6),
            "decode_tokens_per_sec": round(float(tokens_per_sec_steady), 6),
            "latency_ms_total": round(float(steady_total_s) * 1000.0, 3),
            "latency_ms_per_token": round(float(steady_total_s) * 1000.0 / max(1, steady_tokens), 6),
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "vram_peak_mb": None,
            "vram_peak_mb_allocated": None,
            "vram_peak_mb_reserved": None,
            "ram_rss_mb": round(float(_rss_mb() or 0.0), 3) if _rss_mb() is not None else None,
            "ram_peak_mb": round(float(_rss_mb() or 0.0), 3) if _rss_mb() is not None else None,
            "cache_hit_rate": None,
            "bytes_prefetched": None,
            "page_faults": None,
            "page_faults_reason": "mock",
            "paging": {"before": None, "after": None},
            "active_adapters": [],
            "adapter_load_ms": None,
            "adapter_active": None,
            "stream_topk": stream_topk,
        }
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

    model = load_inference_model(settings)
    backend_used = backend
    if bool(getattr(model, "is_hf", False)):
        backend_used = "hf"
    elif bool(getattr(model, "is_llama_cpp", False)):
        backend_used = "llama_cpp"
    try:
        active_adapters, adapter_load_ms, adapter_active = _maybe_prepare_hf_adapters(model, settings, base_dir, prompt_text)
    except Exception:
        active_adapters, adapter_load_ms, adapter_active = [], None, getattr(model, "active_adapter_name", None)
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

    # Prefill/decode estimate: time for 1 token ~= prefill + 1 decode step.
    one_token_s = None
    if int(args.max_new) > 1:
        try:
            start = time.perf_counter()
            _generate(1)
            one_token_s = max(1e-9, time.perf_counter() - start)
        except Exception:
            one_token_s = None

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

    prefill_tps = None
    decode_tps = None
    try:
        repeats = int(max(1, int(args.repeat)))
        avg_full_s = float(steady_total_s) / float(repeats)
        max_new = int(args.max_new)
        if one_token_s is not None and max_new > 1:
            decode_s = float(avg_full_s - float(one_token_s)) / float(max(1, max_new - 1))
            if decode_s <= 1e-12:
                decode_s = float(avg_full_s) / float(max(1, max_new))
            decode_tps = float(1.0 / max(1e-9, decode_s))
            prefill_s = max(0.0, float(one_token_s) - float(decode_s))
            if isinstance(ctx_len_prompt, int) and ctx_len_prompt > 0 and prefill_s > 1e-9:
                prefill_tps = float(ctx_len_prompt) / float(prefill_s)
        else:
            decode_tps = float(tokens_per_sec_steady)
            prefill_tps = None
    except Exception:
        prefill_tps = None
        decode_tps = None

    report: dict[str, Any] = {
        "ok": True,
        "ts": time.time(),
        "profile": str(args.profile),
        "backend": backend_used,
        "stream_topk": stream_topk,
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
        "prefill_tokens_per_sec": round(float(prefill_tps), 6) if prefill_tps is not None else None,
        "decode_tokens_per_sec": round(float(decode_tps), 6) if decode_tps is not None else None,
        "latency_ms_total": round(float(steady_total_s) * 1000.0, 3),
        "latency_ms_per_token": round(float(steady_total_s) * 1000.0 / max(1, steady_tokens), 6),
        "latency_p50_ms": round(float(_pct_ms(steady_s, 50.0) or 0.0), 3) if steady_s else None,
        "latency_p95_ms": round(float(_pct_ms(steady_s, 95.0) or 0.0), 3) if steady_s else None,
        "vram_peak_mb": round(float(vram_peak_allocated_mb), 3) if vram_peak_allocated_mb is not None else None,
        "vram_peak_mb_allocated": round(float(vram_peak_allocated_mb), 3) if vram_peak_allocated_mb is not None else None,
        "vram_peak_mb_reserved": round(float(vram_peak_reserved_mb), 3) if vram_peak_reserved_mb is not None else None,
        "ram_rss_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
        "ram_peak_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
        "active_adapters": list(active_adapters),
        "adapter_load_ms": round(float(adapter_load_ms), 3) if adapter_load_ms is not None else None,
        "adapter_active": adapter_active,
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
