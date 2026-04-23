from __future__ import annotations

import importlib
import importlib.util
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
    scenario: str = "default"


def load_inference_model(*args: Any, **kwargs: Any) -> Any:
    module = importlib.import_module(".model_loader", package=__package__)
    return getattr(module, "load_inference_model")(*args, **kwargs)


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


def _kv_mode_from_settings(settings: dict) -> str:
    runtime = settings.get("runtime", {}) or {}
    raw = str(runtime.get("kv_quant", "none") or "none").strip().lower()
    if raw in {"low_rank", "low-rank", "mla"}:
        return "lowrank"
    return raw


def _kv_telemetry_from_settings(settings: dict) -> dict[str, Any]:
    kv_mode = _kv_mode_from_settings(settings)
    kv_rank = None
    kv_bytes_est = None
    if kv_mode == "lowrank":
        runtime = settings.get("runtime", {}) or {}
        try:
            rank = int(runtime.get("kv_lowrank_rank", 0) or 0)
        except Exception:
            rank = 0
        kv_rank = int(rank) if rank > 0 else None
    try:
        core = settings.get("core", {}) or {}
        vx = settings.get("vortex_model", {}) or {}
        hidden = int(core.get("hidden_size", 0) or 0)
        slots = int(vx.get("latent_slots", vx.get("lava_slots", core.get("lava_slots", 0) or 0)) or 0)
        elem = 4  # float32
        if hidden > 0 and slots > 0:
            if kv_mode == "lowrank":
                rank = int(kv_rank) if isinstance(kv_rank, int) and kv_rank > 0 else max(8, min(256, hidden // 4))
                kv_bytes_est = int(slots) * int(rank) * elem + int(hidden) * int(rank) * elem
            else:
                kv_bytes_est = int(slots) * int(hidden) * elem
    except Exception:
        kv_bytes_est = None
    return {"kv_mode": kv_mode, "kv_rank": kv_rank, "kv_bytes_est": kv_bytes_est}


def _kv_telemetry_from_model(model: Any, settings: dict) -> dict[str, Any]:
    out = _kv_telemetry_from_settings(settings)
    try:
        blocks = getattr(model, "blocks", None)
        if blocks and hasattr(blocks[0], "lava"):
            lava = getattr(blocks[0], "lava")
            mode = str(getattr(lava, "kv_quant", out.get("kv_mode") or "none")).lower()
            if mode in {"low_rank", "low-rank", "mla"}:
                mode = "lowrank"
            out["kv_mode"] = mode
            if mode == "lowrank":
                try:
                    out["kv_rank"] = int(getattr(lava, "kv_lowrank_rank", 0) or 0) or out.get("kv_rank")
                except Exception:
                    pass
            else:
                out["kv_rank"] = None
            bytes_est = None
            if mode == "lowrank" and hasattr(lava, "contents_lr"):
                try:
                    bytes_est = int(lava.contents_lr.numel() * lava.contents_lr.element_size())
                except Exception:
                    bytes_est = None
                try:
                    proj = lava._get_lowrank_projector() if hasattr(lava, "_get_lowrank_projector") else None
                    if proj is not None and hasattr(proj, "proj"):
                        bytes_est = int(bytes_est or 0) + int(proj.proj.numel() * proj.proj.element_size())
                except Exception:
                    pass
            elif hasattr(lava, "contents"):
                try:
                    bytes_est = int(lava.contents.numel() * lava.contents.element_size())
                except Exception:
                    bytes_est = None
            out["kv_bytes_est"] = bytes_est if bytes_est is not None else out.get("kv_bytes_est")
    except Exception:
        return out
    return out


def _resolve_shared_expert_cfg(settings: dict, base_dir: Path) -> dict[str, Any] | None:
    for section in ("experts", "adapters"):
        cfg = settings.get(section, {}) or {}
        raw_path = cfg.get("shared_expert_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            continue
        raw_name = cfg.get("shared_expert_name") or "shared_expert"
        raw_weight = cfg.get("shared_expert_weight")
        try:
            weight = float(raw_weight) if raw_weight is not None else 0.2
        except Exception:
            weight = 0.2
        if weight < 0:
            weight = 0.0
        return {"section": section, "name": str(raw_name), "path": str(path), "weight": float(weight)}
    return None


def _maybe_prepare_hf_adapters(
    model: Any, settings: dict, base_dir: Path, prompt: str
) -> tuple[list[str], float | None, str | None, dict[str, Any] | None]:
    """Best-effort: load+activate adapters for HF bench.

    Returns (active_adapters, load_ms, active_adapter_name, telemetry).
    """
    if not bool(getattr(model, "is_hf", False)):
        return [], None, None, None

    registry = None
    router = None
    chosen_section = None
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
                chosen_section = str(section)
                break
        except Exception:
            registry = None
            router = None
            chosen_section = None
            continue
    if registry is None or router is None:
        return [], None, None, None

    decision = router.select(prompt or "", registry.names, top_k=None)
    selected: list[str] = []
    if decision.selected_adapters:
        selected = [str(x).strip() for x in decision.selected_adapters if str(x).strip()]
    elif decision.selected_adapter:
        selected = [str(decision.selected_adapter).strip()]
    selected = [name for name in selected if name]
    if not selected:
        return [], None, None, {"selected_adapters": [], "shared_expert": None, "shared_used": False}

    if not hasattr(model, "add_adapter"):
        return [], None, getattr(model, "active_adapter_name", None), {"selected_adapters": list(selected), "shared_expert": None, "shared_used": False}

    mix_mode = _resolve_mix_mode(settings)
    shared_cfg = _resolve_shared_expert_cfg(settings, base_dir=base_dir)
    if shared_cfg is not None and chosen_section and str(shared_cfg.get("section")) != str(chosen_section):
        shared_cfg = None
    if shared_cfg is not None and hasattr(model, "set_weighted_adapters"):
        mix_mode = "weighted"
    start = time.perf_counter()
    loaded: list[str] = []
    cache_hits = 0
    shared_name = str(shared_cfg.get("name")) if shared_cfg else None
    shared_path = str(shared_cfg.get("path")) if shared_cfg else None
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
                loaded_new = bool(model.add_adapter(name, path))
                if not loaded_new:
                    cache_hits += 1
                loaded.append(name)
            except Exception:
                continue
        if shared_name and shared_path and shared_name not in loaded and Path(shared_path).exists():
            try:
                loaded_new = bool(model.add_adapter(shared_name, shared_path))
                if not loaded_new:
                    cache_hits += 1
                loaded.append(shared_name)
            except Exception:
                pass

        if not loaded:
            return [], round((time.perf_counter() - start) * 1000.0, 3), getattr(model, "active_adapter_name", None), {"selected_adapters": list(selected), "shared_expert": None, "shared_used": False}

        # Activate selection (single or weighted mix).
        shared_used = False
        weights_out = None
        if mix_mode == "weighted" and len(loaded) > 1 and hasattr(model, "set_weighted_adapters"):
            base_names = [n for n in selected if n in loaded]
            base_scores = decision.scores if isinstance(decision.scores, list) and len(decision.scores) == len(selected) else None
            weights = _normalize_weights(base_scores) if base_scores else ([1.0 / float(len(base_names)) for _ in base_names] if base_names else [])
            adapter_weights = {name: float(weight) for name, weight in zip(base_names, weights)}
            if shared_name and shared_path and shared_name in loaded:
                try:
                    shared_w = float(shared_cfg.get("weight", 0.2)) if shared_cfg else 0.2
                except Exception:
                    shared_w = 0.2
                adapter_weights[shared_name] = float(shared_w)
            total = float(sum(max(0.0, float(v)) for v in adapter_weights.values()))
            if total > 1e-12:
                adapter_weights = {k: float(v) / total for k, v in adapter_weights.items()}
            try:
                mixed_ok = bool(model.set_weighted_adapters(adapter_weights))
                shared_used = bool(mixed_ok and shared_name and shared_name in adapter_weights)
                weights_out = dict(adapter_weights)
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
    telemetry = {
        "selected_adapters": list(selected),
        "active_adapters": list(loaded),
        "shared_expert": (shared_name if shared_name and shared_name in loaded else None),
        "shared_used": bool(shared_used),
        "adapter_cache_hit": int(cache_hits),
        "weights": weights_out,
        "mix_mode": mix_mode,
    }
    return loaded, float(elapsed_ms), getattr(model, "active_adapter_name", None), telemetry


def _rss_mb() -> float | None:
    try:
        proc = psutil.Process()
        return float(proc.memory_info().rss) / 1e6
    except Exception:
        return None


def _proc_mem_mb() -> dict[str, float | None]:
    try:
        proc = psutil.Process()
        info = proc.memory_info()
        rss_mb = float(getattr(info, "rss", 0.0) or 0.0) / 1e6
        vms_mb = float(getattr(info, "vms", 0.0) or 0.0) / 1e6
        commit_mb = None
        try:
            full = proc.memory_full_info()
            private_bytes = getattr(full, "private", None)
            uss_bytes = getattr(full, "uss", None)
            commit_bytes = private_bytes if private_bytes is not None else uss_bytes
            if commit_bytes is not None:
                commit_mb = float(commit_bytes) / 1e6
        except Exception:
            commit_mb = None
        if commit_mb is None:
            commit_mb = vms_mb
        return {"rss_mb": rss_mb, "vms_mb": vms_mb, "commit_mb": commit_mb}
    except Exception:
        return {"rss_mb": None, "vms_mb": None, "commit_mb": None}


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
    scenario = str(getattr(args, "scenario", "") or "default")
    backend_resolved = backend
    quant_mode = None
    offload_enabled = None
    try:
        from .prepare import prepare_model_state

        prep = prepare_model_state(settings, base_dir=base_dir)
        if isinstance(prep, dict):
            backend_resolved = str(prep.get("backend_resolved") or backend_resolved).lower()
            quant_mode = prep.get("quant_mode")
            offload_enabled = prep.get("offload_enabled")
    except Exception:
        prep = None
        backend_resolved = backend
        quant_requested = bool(core_cfg.get("hf_load_in_4bit") or core_cfg.get("hf_load_in_8bit"))
        bnb = importlib.util.find_spec("bitsandbytes") is not None
        if backend in {"hf", "transformers"} and quant_requested and bnb:
            quant_mode = "4bit" if bool(core_cfg.get("hf_load_in_4bit")) else "8bit"
        if backend in {"hf", "transformers"}:
            offload_enabled = bool(core_cfg.get("hf_device_map") or core_cfg.get("hf_max_memory") or core_cfg.get("hf_offload_folder"))
        if backend == "llama_cpp":
            quant_mode = "gguf"
            offload_enabled = False

    engine = str(core_cfg.get("engine") or core_cfg.get("external_engine") or backend_resolved).strip().lower() or backend_resolved
    kv_tel = _kv_telemetry_from_settings(settings)
    shared_expert = None

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
        per_iter_s = float(steady_total_s) / float(max(1, int(args.repeat) or 1))
        steady_s = [per_iter_s for _ in range(max(1, int(args.repeat) or 1))]
        mem = _proc_mem_mb()
        report: dict[str, Any] = {
            "ok": True,
            "ts": time.time(),
            "profile": str(args.profile),
            "backend": backend,
            "backend_resolved": backend_resolved,
            "engine": engine,
            "quant_mode": quant_mode,
            "offload_enabled": offload_enabled,
            "scenario": scenario,
            **kv_tel,
            "shared_expert": shared_expert,
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
            "latency_p50_ms": round(float(_pct_ms(steady_s, 50.0) or 0.0), 3),
            "latency_p95_ms": round(float(_pct_ms(steady_s, 95.0) or 0.0), 3),
            "vram_peak_mb": None,
            "vram_peak_mb_allocated": None,
            "vram_peak_mb_reserved": None,
            "rss_mb": round(float(mem.get("rss_mb") or 0.0), 3) if mem.get("rss_mb") is not None else None,
            "commit_mb": round(float(mem.get("commit_mb") or 0.0), 3) if mem.get("commit_mb") is not None else None,
            "ram_rss_mb": round(float(mem.get("rss_mb") or 0.0), 3) if mem.get("rss_mb") is not None else None,
            "ram_commit_mb": round(float(mem.get("commit_mb") or 0.0), 3) if mem.get("commit_mb") is not None else None,
            "ram_peak_mb": round(float(mem.get("rss_mb") or 0.0), 3) if mem.get("rss_mb") is not None else None,
            "cache_hit_rate": None,
            "bytes_prefetched": None,
            "page_faults": None,
            "page_faults_reason": "mock",
            "paging": {"before": None, "after": None},
            "active_adapters": [],
            "adapter_load_ms": None,
            "adapter_active": None,
            "adapter_cache_hit": None,
            "selected_adapters": None,
            "shared_used": None,
            "adapter_weights": None,
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
    backend_resolved = backend_used
    if bool(getattr(model, "is_hf", False)):
        try:
            cfg = getattr(model, "cfg", None)
            load_kwargs = getattr(cfg, "load_kwargs", None) if cfg is not None else None
            if isinstance(load_kwargs, dict):
                if bool(load_kwargs.get("load_in_4bit")):
                    quant_mode = "4bit"
                elif bool(load_kwargs.get("load_in_8bit")):
                    quant_mode = "8bit"
                offload_enabled = bool(load_kwargs.get("device_map") or load_kwargs.get("max_memory") or load_kwargs.get("offload_folder"))
        except Exception:
            pass
    elif bool(getattr(model, "is_llama_cpp", False)):
        quant_mode = "gguf"
        offload_enabled = False
    kv_tel = _kv_telemetry_from_model(model, settings)
    adapter_tel = None
    try:
        active_adapters, adapter_load_ms, adapter_active, adapter_tel = _maybe_prepare_hf_adapters(model, settings, base_dir, prompt_text)
    except Exception:
        active_adapters, adapter_load_ms, adapter_active, adapter_tel = [], None, getattr(model, "active_adapter_name", None), None
    if isinstance(adapter_tel, dict):
        shared_expert = adapter_tel.get("shared_expert")
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

    mem = _proc_mem_mb()
    rss_mb = mem.get("rss_mb")

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
        "backend_resolved": backend_resolved,
        "engine": engine,
        "quant_mode": quant_mode,
        "offload_enabled": offload_enabled,
        "scenario": scenario,
        **kv_tel,
        "shared_expert": shared_expert,
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
        "rss_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
        "commit_mb": round(float(mem.get("commit_mb") or 0.0), 3) if mem.get("commit_mb") is not None else None,
        "ram_rss_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
        "ram_commit_mb": round(float(mem.get("commit_mb") or 0.0), 3) if mem.get("commit_mb") is not None else None,
        "ram_peak_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
        "active_adapters": list(active_adapters),
        "adapter_load_ms": round(float(adapter_load_ms), 3) if adapter_load_ms is not None else None,
        "adapter_active": adapter_active,
        "adapter_cache_hit": (adapter_tel.get("adapter_cache_hit") if isinstance(adapter_tel, dict) else None),
        "selected_adapters": (adapter_tel.get("selected_adapters") if isinstance(adapter_tel, dict) else None),
        "shared_used": (adapter_tel.get("shared_used") if isinstance(adapter_tel, dict) else None),
        "adapter_weights": (adapter_tel.get("weights") if isinstance(adapter_tel, dict) else None),
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
