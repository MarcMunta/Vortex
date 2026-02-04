from __future__ import annotations

import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def resolve_profile(profile: str | None = None) -> str:
    env_profile = os.getenv("C3RNT2_PROFILE")
    return profile or env_profile or "dev_small"


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_profile(profiles: dict[str, Any], name: str, stack: list[str]) -> dict[str, Any]:
    if name in stack:
        cycle = " -> ".join(stack + [name])
        raise ValueError(f"base_profile cycle detected: {cycle}")
    if name not in profiles:
        raise KeyError(f"Profile '{name}' not found")
    profile = profiles[name]
    base_name = profile.get("base_profile")
    if base_name:
        base_profile = _resolve_profile(profiles, base_name, stack + [name])
        override = {k: v for k, v in profile.items() if k != "base_profile"}
        return _merge_dicts(base_profile, override)
    return deepcopy(profile)




def normalize_settings(settings: dict) -> dict:
    normalized = deepcopy(settings)
    tok = normalized.get("tokenizer", {}) or {}
    if "vortex_tok_path" not in tok and tok.get("vortex_model_path"):
        tok["vortex_tok_path"] = tok.get("vortex_model_path")
    normalized["tokenizer"] = tok

    runtime = normalized.get("runtime")
    c3 = normalized.get("c3")
    if runtime is None:
        runtime = {}
    runtime = dict(runtime)
    if c3:
        runtime.setdefault("paged_lm_head", True)
        runtime.setdefault("paged_tile_out", c3.get("tile_size"))
        runtime.setdefault("paged_tile_in", c3.get("tile_in"))
        runtime.setdefault("cache_vram_budget_mb", c3.get("cache_vram_budget_mb"))
        runtime.setdefault("paged_lm_head_stream_topk", c3.get("paged_lm_head_stream_topk"))
        runtime.setdefault("prefetch_depth", c3.get("prefetch_depth"))
        runtime.setdefault("compression", c3.get("compression"))
        runtime.setdefault("pinned_memory", c3.get("pinned_memory"))
    if "paged_lm_head" not in runtime:
        runtime["paged_lm_head"] = False
    if "cache_vram_budget_mb" not in runtime:
        runtime["cache_vram_budget_mb"] = 2048
    runtime.setdefault("prefetch_depth", 2)
    runtime.setdefault("paged_lm_head_stream_topk", runtime.get("paged_lm_head_stream_topk", False) or False)
    kv = normalized.get("kv", {}) or {}
    if "kv_quant" not in runtime:
        kv_bits = kv.get("kv_quant_bits")
        if kv_bits is not None:
            if int(kv_bits) == 8:
                runtime["kv_quant"] = "int8"
            elif int(kv_bits) == 2:
                runtime["kv_quant"] = "2bit"
            elif int(kv_bits) <= 0:
                runtime["kv_quant"] = "none"
    runtime.setdefault("kv_quant", "none")
    runtime.setdefault("kv_quant_2bit_experimental", False)
    runtime.setdefault("i_know_what_im_doing", False)
    runtime.setdefault("gpu_decompress", "none")
    normalized["runtime"] = runtime

    tools = normalized.get("tools", {}) or {}
    web = tools.get("web", {}) or {}
    agent = normalized.get("agent", {}) or {}
    if not web.get("allow_domains") and agent.get("web_allowlist"):
        web["allow_domains"] = agent.get("web_allowlist")
    web.setdefault("enabled", False)
    web.setdefault("allow_domains", ["docs.python.org", "pytorch.org", "github.com"])
    web.setdefault("search_domains", ["duckduckgo.com"])
    web.setdefault("max_bytes", 512000)
    web.setdefault("timeout_s", 10)
    web.setdefault("rate_limit_per_min", agent.get("rate_limit_per_min", 30))
    web.setdefault("cache_dir", "data/web_cache")
    web.setdefault("cache_ttl_s", 3600)
    web.setdefault("allow_content_types", ["text/", "application/json"])
    tools["web"] = web
    normalized["tools"] = tools

    security = normalized.get("security", {}) or {}
    security = dict(security) if isinstance(security, dict) else {}
    web_sec = security.get("web", {}) or {}
    web_sec = dict(web_sec) if isinstance(web_sec, dict) else {}
    web_sec.setdefault("strict", True)
    web_sec.setdefault("allowlist_domains", None)
    security["web"] = web_sec
    normalized["security"] = security

    self_patch = normalized.get("self_patch", {}) or {}
    self_patch.setdefault("enabled", False)
    self_patch.setdefault("auto_sandbox", True)
    self_patch.setdefault("queue_dir", "data/self_patch/queue")
    self_patch.setdefault("sandbox_dir", "data/self_patch/sandbox")
    self_patch.setdefault("max_patch_kb", 128)
    self_patch.setdefault("allowed_paths", ["src/", "tests/"])
    self_patch.setdefault("run_tests_on_apply", True)
    self_patch.setdefault("allowed_commands", ["pytest", "ruff", "python"])
    self_patch.setdefault(
        "forbidden_globs",
        [
            ".env",
            ".env.*",
            "data/**",
            "*.key",
            "*.pem",
            "*.p12",
            "*.sqlite",
            "*.db",
            "keys/**",
            "secrets/**",
            "src/c3rnt2/self_patch/**",
            "src/c3rnt2/selfimprove/**",
        ],
    )
    normalized["self_patch"] = self_patch

    agent = normalized.get("agent", {}) or {}
    agent.setdefault("max_iters", 5)
    agent.setdefault(
        "tools_enabled",
        [
            "open_docs",
            "search_web",
            "read_file",
            "grep",
            "list_tree",
            "run_tests",
            "propose_patch",
            "sandbox_patch",
            "apply_patch",
            "summarize_diff",
        ],
    )
    normalized["agent"] = agent

    server_cfg = normalized.get("server", {}) or {}
    server_cfg.setdefault("auto_reload_adapter", False)
    server_cfg.setdefault("reload_interval_s", 60)
    server_cfg.setdefault("reload_request_interval_s", 2)
    server_cfg.setdefault("maintenance_window_s", 10)
    server_cfg.setdefault("block_during_training", False)
    server_cfg.setdefault("train_strategy", "subprocess")
    normalized["server"] = server_cfg

    knowledge = normalized.get("knowledge", {}) or {}
    knowledge.setdefault("embedding_backend", "auto")
    knowledge.setdefault("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    knowledge.setdefault("index_backend", "auto")
    policy = knowledge.get("policy", {}) or {}
    policy.setdefault("min_quality", 0.0)
    policy.setdefault("max_age_days", None)
    policy.setdefault("allow_domains", None)
    policy.setdefault("deny_domains", None)
    policy.setdefault("allow_source_kinds", None)
    policy.setdefault("deny_source_kinds", None)
    knowledge["policy"] = policy
    normalized["knowledge"] = knowledge

    adapters = normalized.get("adapters", {}) or {}
    adapters.setdefault("enabled", False)
    adapters.setdefault("paths", {})
    adapters.setdefault("max_loaded", 0)
    adapters.setdefault("default", None)
    adapter_router = adapters.get("router", {}) or {}
    adapter_router.setdefault("mode", "keyword_map")
    adapter_router.setdefault("keyword_map", {})
    adapter_router.setdefault("default", adapters.get("default"))
    adapter_router.setdefault("embedding_backend", knowledge.get("embedding_backend", "hash"))
    adapter_router.setdefault("embedding_dim", 128)
    adapter_router.setdefault("embedding_min_score", 0.0)
    adapters["router"] = adapter_router
    normalized["adapters"] = adapters

    hf_train = normalized.get("hf_train", {}) or {}
    hf_train.setdefault("enabled", False)
    core_ref = normalized.get("core", {}) or {}
    if not hf_train.get("model_name") and core_ref.get("hf_model"):
        hf_train["model_name"] = core_ref.get("hf_model")
    hf_train.setdefault("registry_dir", "data/registry/hf_train")
    hf_train.setdefault("dataset_path", "data/registry/hf_train/sft_samples.jsonl")
    hf_train.setdefault("state_path", "data/registry/hf_train/state.json")
    hf_train.setdefault("max_samples", 128)
    hf_train.setdefault("min_quality", 0.0)
    hf_train.setdefault("prompt_template", "Context:\n{text}\nAnswer:")
    hf_train.setdefault("max_seq_len", 1024)
    hf_train.setdefault("micro_batch_size", 1)
    hf_train.setdefault("grad_accum_steps", 4)
    hf_train.setdefault("max_steps", 50)
    hf_train.setdefault("lr", 2e-4)
    hf_train.setdefault("auto_tune_batch", True)
    hf_train.setdefault("auto_tune_retries", 2)
    hf_train.setdefault("load_in_4bit", True)
    hf_train.setdefault("load_in_8bit", False)
    hf_train.setdefault("lora_rank", 8)
    hf_train.setdefault("lora_alpha", 16)
    hf_train.setdefault("lora_dropout", 0.05)
    hf_train.setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    hf_train.setdefault("min_chars", 40)
    hf_train.setdefault("max_repeat_ratio", 0.8)
    hf_train.setdefault("semantic_dedup_threshold", 0.97)
    hf_train.setdefault("include_soft_feedback", True)
    hf_train.setdefault("pack_samples", False)
    hf_train.setdefault("bucket_by_length", True)
    hf_train.setdefault("grad_clip", 1.0)
    hf_train.setdefault("use_weighted_sampling", True)
    hf_train.setdefault(
        "source_kind_weights",
        {
            "chat_feedback": 1.5,
            "chat_feedback_soft": 0.7,
            "episode": 1.2,
            "web": 0.8,
            "logs": 0.8,
        },
    )
    hf_eval = hf_train.get("eval", {}) or {}
    hf_eval.setdefault("enabled", True)
    hf_eval.setdefault("min_improvement", 0.0)
    hf_eval.setdefault("max_regression", 0.0)
    hf_eval.setdefault("max_samples", 8)
    hf_eval.setdefault("gen_max_new_tokens", 64)
    hf_eval.setdefault("max_repeat_ratio", 0.9)
    hf_train["eval"] = hf_eval
    normalized["hf_train"] = hf_train

    learning = normalized.get("learning", {}) or {}
    learning.setdefault("raw_path", "data/learning/raw.jsonl")
    learning.setdefault("curated_path", "data/learning/curated.jsonl")
    learning.setdefault("state_path", "data/learning/state.sqlite")
    learning.setdefault("evals_path", "data/learning/evals.jsonl")
    learning.setdefault("canary_path", "data/learning/canary.jsonl")
    learning.setdefault("max_events", 500)
    learning.setdefault("min_chars", 20)
    learning.setdefault("max_chars", None)
    learning.setdefault("max_eval_samples", 8)
    learning.setdefault("promote_min_improvement", 0.0)
    learning.setdefault("require_eval_ok", True)
    learning.setdefault("require_bench_ok", False)
    normalized["learning"] = learning

    vx = normalized.get("vortex_model", {}) or {}
    core = normalized.get("core", {}) or {}
    core.setdefault("backend", "vortex")
    core.setdefault("vram_threshold_mb", 1024)
    core.setdefault("vram_floor_tokens", 32)
    core.setdefault("vram_ceil_tokens", 512)
    core.setdefault("vram_safety_margin_mb", 512)
    if "tf32" not in core and core.get("allow_tf32") is not None:
        core["tf32"] = core.get("allow_tf32")
    normalized["core"] = core
    lava_keys = {
        "lava_top_k",
        "lava_clusters",
        "lava_cluster_top",
        "lava_read_every",
        "lava_write_every",
        "lava_write_on_surprise",
        "lava_surprise_threshold",
        "lava_cluster_ema",
        "lava_cluster_reassign_threshold",
        "lava_ann_mode",
        "lava_shared_groups",
    }
    lava = {}
    for key in lava_keys:
        if key in vx:
            lava[key] = vx.get(key)
        elif key in core:
            lava[key] = core.get(key)
    cont = normalized.get("continuous", {}) or {}
    if cont:
        if cont.get("run_interval_minutes") is not None:
            warnings.warn("continuous.run_interval_minutes is deprecated; use continuous.interval_minutes", DeprecationWarning)
            cont.setdefault("interval_minutes", cont.get("run_interval_minutes"))
        if "run_interval_minutes" not in cont and cont.get("interval_minutes") is not None:
            cont["run_interval_minutes"] = cont.get("interval_minutes")
        if "max_steps_per_tick" not in cont and cont.get("max_steps") is not None:
            cont["max_steps_per_tick"] = cont.get("max_steps")
        if "max_steps" not in cont and cont.get("max_steps_per_tick") is not None:
            cont["max_steps"] = cont.get("max_steps_per_tick")
        web_disc = cont.get("web_discovery", {}) or {}
        web_disc.setdefault("enabled", False)
        web_disc.setdefault("seed_queries", [])
        web_disc.setdefault("max_urls_per_tick", 10)
        web_disc.setdefault("max_total_urls", 200)
        web_disc.setdefault("ttl_hours", 72)
        web_disc.setdefault("max_queue", 500)
        web_disc.setdefault("max_crawl_pages_per_tick", 2)
        web_disc.setdefault("max_links_per_page", 50)
        web_disc.setdefault("max_sitemap_urls", 200)
        cont["web_discovery"] = web_disc
        normalized["continuous"] = cont

    autopilot = normalized.get("autopilot", {}) or {}
    autopilot.setdefault("enabled", False)
    autopilot.setdefault("interval_minutes", cont.get("interval_minutes", 30) if cont else 30)
    autopilot.setdefault("ingest_cooldown_minutes", 10)
    autopilot.setdefault("train_cooldown_minutes", 60)
    autopilot.setdefault("eval_cooldown_minutes", 60)
    autopilot.setdefault("patch_cooldown_minutes", 120)
    autopilot.setdefault("train_max_steps", hf_train.get("max_steps", 50))
    autopilot.setdefault("training_jsonl_max_items", 500)
    autopilot.setdefault("min_improvement", hf_eval.get("min_improvement", 0.0))
    autopilot.setdefault("reuse_dataset", False)
    autopilot.setdefault("autopatch_enabled", False)
    autopilot.setdefault("autopatch_on_test_fail", True)
    autopilot.setdefault("autopatch_on_doctor_fail", True)
    autopilot.setdefault("autopatch_require_eval", True)
    autopilot.setdefault("autopatch_strategy", "subprocess_cpu")
    autopilot.setdefault("autopatch_require_approval", False)
    autopilot.setdefault("approval_file", "data/APPROVE_AUTOPATCH")
    autopilot.setdefault("restart_after_patch", False)
    autopilot.setdefault("bench_enabled", False)
    autopilot.setdefault("bench_max_new_tokens", 64)
    autopilot.setdefault("bench_max_regression", 0.15)
    autopilot.setdefault("bench_min_tokens_per_sec", 0.0)
    # Disabled by default; autonomous profiles should opt in.
    autopilot.setdefault("min_new_samples_per_tick", 0)
    autopilot.setdefault("max_consecutive_failures", 3)
    autopilot.setdefault("safe_mode_cooldown_minutes", 0)
    autopilot.setdefault("todo_regex", r"TODO\((P1|PRIORITY)\)|TODO!|TODO:HIGH|TODO:CRITICAL")
    normalized["autopilot"] = autopilot

    if lava:
        normalized["lava"] = lava

    return normalized


def resolve_web_allowlist(settings: dict) -> list[str]:
    security = settings.get("security", {}) or {}
    web_sec = security.get("web", {}) or {}
    allowlist_domains = web_sec.get("allowlist_domains")
    if isinstance(allowlist_domains, list):
        return [str(item) for item in allowlist_domains if item]
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    if isinstance(web_cfg.get("allow_domains"), list) and web_cfg.get("allow_domains"):
        return [str(item) for item in web_cfg.get("allow_domains") if item]
    agent_cfg = settings.get("agent", {}) or {}
    return [str(item) for item in agent_cfg.get("web_allowlist", []) if item]


def resolve_web_strict(settings: dict) -> bool:
    security = settings.get("security", {}) or {}
    web_sec = security.get("web", {}) or {}
    strict = web_sec.get("strict")
    if strict is None:
        return True
    return bool(strict)


def _get_nested(d: dict, keys: list[str], default: object = None) -> object:
    cur: object = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _set_nested(d: dict, keys: list[str], value: object) -> None:
    cur = d
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _clamp_int(settings: dict, keys: list[str], *, max_value: int, label: str) -> None:
    raw = _get_nested(settings, keys, None)
    if raw is None:
        return
    try:
        val = int(raw)
    except Exception:
        return
    if val > int(max_value):
        warnings.warn(f"{label} clamped to {int(max_value)} for rtx4080_16gb_safe (was {val})")
        _set_nested(settings, keys, int(max_value))


def _apply_rtx4080_16gb_safe_clamps(settings: dict) -> dict:
    # Hard safety clamps for RTX 4080 16GB profiles. These are NOT usage quotas;
    # they prevent runaway VRAM/ctx/cache settings that can OOM the process.
    profile = str(settings.get("_profile") or "")
    if profile != "rtx4080_16gb_safe":
        return settings

    _clamp_int(settings, ["decode", "max_new_tokens"], max_value=128, label="decode.max_new_tokens")
    _clamp_int(settings, ["core", "local_window"], max_value=2048, label="core.local_window")
    _clamp_int(settings, ["kv", "window_size"], max_value=2048, label="kv.window_size")
    _clamp_int(settings, ["runtime", "cache_vram_budget_mb"], max_value=4096, label="runtime.cache_vram_budget_mb")
    _clamp_int(settings, ["c3", "cache_vram_budget_mb"], max_value=4096, label="c3.cache_vram_budget_mb")
    _clamp_int(settings, ["runtime", "prefetch_depth"], max_value=4, label="runtime.prefetch_depth")
    _clamp_int(settings, ["continuous", "batch_tokens"], max_value=8192, label="continuous.batch_tokens")

    runtime = dict(settings.get("runtime", {}) or {})
    if str(runtime.get("kv_quant_2bit_experimental", "")).lower() not in {"1", "true", "yes"}:
        runtime["kv_quant_2bit_experimental"] = False
    settings["runtime"] = runtime
    return settings



def validate_profile(settings: dict, base_dir: Path | None = None) -> None:
    missing: list[str] = []
    errors: list[str] = []
    base_dir = Path(base_dir or ".").resolve()
    tok = settings.get("tokenizer", {}) or {}
    core = settings.get("core", {}) or {}
    backend = str(core.get("backend", "vortex")).lower()
    runtime = settings.get("runtime", {}) or {}
    decode = settings.get("decode", {}) or {}
    bad = settings.get("bad", {}) or {}
    cont = settings.get("continuous", {}) or {}
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    self_patch_cfg = settings.get("self_patch", {}) or {}
    hf_train_cfg = settings.get("hf_train", {}) or {}
    adapters_cfg = settings.get("adapters", {}) or {}

    if not tok.get("vortex_tok_path"):
        missing.append("tokenizer.vortex_tok_path")
    if backend == "hf":
        if not core.get("hf_model"):
            missing.append("core.hf_model")
    elif backend == "tensorrt":
        if not (core.get("tensorrt_engine_dir") or core.get("tensorrt_engine_path")):
            missing.append("core.tensorrt_engine_dir")
        if not (core.get("tensorrt_tokenizer") or core.get("hf_model")):
            missing.append("core.tensorrt_tokenizer or core.hf_model")
    else:
        for key in ("hidden_size", "layers", "heads"):
            if key not in core:
                missing.append(f"core.{key}")

    if "cache_vram_budget_mb" not in runtime:
        missing.append("runtime.cache_vram_budget_mb")
    else:
        if float(runtime.get("cache_vram_budget_mb", 0)) <= 0:
            errors.append("runtime.cache_vram_budget_mb must be > 0")
    stream_topk = runtime.get("paged_lm_head_stream_topk")
    if stream_topk is not None and stream_topk is not False:
        if int(stream_topk) <= 0:
            errors.append("runtime.paged_lm_head_stream_topk must be > 0")
    prefetch_depth = runtime.get("prefetch_depth")
    if prefetch_depth is not None and int(prefetch_depth) < 0:
        errors.append("runtime.prefetch_depth must be >= 0")

    kv_quant = str(runtime.get("kv_quant", "none")).lower()
    if kv_quant not in {"none", "int8", "2bit"}:
        errors.append("runtime.kv_quant must be one of none|int8|2bit")
    if kv_quant == "2bit":
        if not bool(runtime.get("kv_quant_2bit_experimental", False)):
            errors.append("runtime.kv_quant=2bit is experimental; set runtime.kv_quant_2bit_experimental=true")
        if not bool(runtime.get("i_know_what_im_doing", False)):
            errors.append("runtime.kv_quant=2bit requires runtime.i_know_what_im_doing=true")
    gpu_decompress = str(runtime.get("gpu_decompress", "none")).lower()
    if gpu_decompress not in {"none", "triton"}:
        errors.append("runtime.gpu_decompress must be none or triton (CPU decompress + H2D pipeline)")

    if web_cfg:
        if bool(web_cfg.get("enabled", False)) and not web_cfg.get("allow_domains"):
            errors.append("tools.web.allow_domains required when tools.web.enabled is true")
        if bool(web_cfg.get("enabled", False)) and not web_cfg.get("allow_content_types"):
            errors.append("tools.web.allow_content_types required when tools.web.enabled is true")
        try:
            if int(web_cfg.get("rate_limit_per_min", 1)) <= 0:
                errors.append("tools.web.rate_limit_per_min must be > 0")
        except Exception:
            errors.append("tools.web.rate_limit_per_min must be > 0")
        try:
            if int(web_cfg.get("max_bytes", 1)) <= 0:
                errors.append("tools.web.max_bytes must be > 0")
        except Exception:
            errors.append("tools.web.max_bytes must be > 0")
        try:
            if float(web_cfg.get("timeout_s", 1.0)) <= 0:
                errors.append("tools.web.timeout_s must be > 0")
        except Exception:
            errors.append("tools.web.timeout_s must be > 0")

    if self_patch_cfg:
        if not self_patch_cfg.get("allowed_paths"):
            errors.append("self_patch.allowed_paths must not be empty")
        try:
            if int(self_patch_cfg.get("max_patch_kb", 1)) <= 0:
                errors.append("self_patch.max_patch_kb must be > 0")
        except Exception:
            errors.append("self_patch.max_patch_kb must be > 0")

    if hf_train_cfg and bool(hf_train_cfg.get("enabled", False)):
        if not (hf_train_cfg.get("model_name") or core.get("hf_model")):
            errors.append("hf_train.model_name or core.hf_model required for hf training")
        try:
            if int(hf_train_cfg.get("micro_batch_size", 1)) <= 0:
                errors.append("hf_train.micro_batch_size must be > 0")
        except Exception:
            errors.append("hf_train.micro_batch_size must be > 0")
        try:
            if int(hf_train_cfg.get("grad_accum_steps", 1)) <= 0:
                errors.append("hf_train.grad_accum_steps must be > 0")
        except Exception:
            errors.append("hf_train.grad_accum_steps must be > 0")

    if adapters_cfg and bool(adapters_cfg.get("enabled", False)):
        paths = adapters_cfg.get("paths", {}) or {}
        if not paths:
            errors.append("adapters.paths must not be empty when adapters.enabled is true")
        router_cfg = adapters_cfg.get("router", {}) or {}
        keyword_map = router_cfg.get("keyword_map", {}) or {}
        for _kw, name in keyword_map.items():
            if name and name not in paths:
                errors.append(f"adapters.router.keyword_map references unknown adapter: {name}")
        default = adapters_cfg.get("default") or router_cfg.get("default")
        if default and default not in paths:
            errors.append(f"adapters.default unknown: {default}")
        try:
            if int(hf_train_cfg.get("max_steps", 1)) <= 0:
                errors.append("hf_train.max_steps must be > 0")
        except Exception:
            errors.append("hf_train.max_steps must be > 0")
        try:
            if float(hf_train_cfg.get("lr", 1e-6)) <= 0:
                errors.append("hf_train.lr must be > 0")
        except Exception:
            errors.append("hf_train.lr must be > 0")

    top_p = float(decode.get("top_p", bad.get("top_p", 1.0)))
    if not (0.0 < top_p <= 1.0):
        errors.append("decode.top_p must be in (0, 1]")
    top_p_min_k = int(bad.get("top_p_min_k", decode.get("top_p_min_k", 0)) or 0)
    top_p_max_k = int(bad.get("top_p_max_k", decode.get("top_p_max_k", 0)) or 0)
    if top_p_min_k and top_p_max_k and top_p_min_k > top_p_max_k:
        errors.append("top_p_min_k must be <= top_p_max_k")
    draft_cfg = decode.get("draft_model", {}) or {}
    if draft_cfg.get("enabled"):
        draft_layers = int(draft_cfg.get("draft_layers", 0))
        if draft_layers <= 0:
            errors.append("decode.draft_model.draft_layers must be > 0")
        core_layers = int(core.get("layers", 0))
        if core_layers and draft_layers > core_layers:
            errors.append("decode.draft_model.draft_layers must be <= core.layers")

    interval = cont.get("interval_minutes", cont.get("run_interval_minutes"))
    if interval is not None and float(interval) <= 0:
        errors.append("continuous.interval_minutes must be > 0")
    max_steps = cont.get("max_steps_per_tick", cont.get("max_steps"))
    if max_steps is not None and int(max_steps) <= 0:
        errors.append("continuous.max_steps_per_tick must be > 0")
    lr = cont.get("lr")
    if lr is not None and float(lr) <= 0:
        errors.append("continuous.lr must be > 0")
    batch_tokens = cont.get("batch_tokens")
    if batch_tokens is not None and int(batch_tokens) <= 0:
        errors.append("continuous.batch_tokens must be > 0")

    tools = settings.get("tools", {}) or {}
    web = tools.get("web", {}) or {}
    if web.get("enabled"):
        allow = web.get("allow_domains", [])
        if not allow:
            errors.append("tools.web.allow_domains required when web enabled")
        allow_types = web.get("allow_content_types", [])
        if not allow_types:
            errors.append("tools.web.allow_content_types required when web enabled")

    self_patch = settings.get("self_patch", {}) or {}
    if self_patch.get("enabled"):
        allowed_paths = self_patch.get("allowed_paths", [])
        if not allowed_paths:
            errors.append("self_patch.allowed_paths required when self_patch enabled")

    learning = settings.get("learning", {}) or {}

    data_root = (base_dir / "data").resolve()

    def _check_data_path(path_value: str | Path | None, label: str) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if data_root not in path.parents and path != data_root:
            errors.append(f"{label} must be under ./data")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"{label} parent not writable: {exc}")
            return
        if not os.access(path.parent, os.W_OK):
            errors.append(f"{label} parent not writable")

    _check_data_path(cont.get("knowledge_path"), "continuous.knowledge_path")
    _check_data_path(cont.get("replay", {}).get("path"), "continuous.replay.path")
    _check_data_path(cont.get("eval", {}).get("anchors_path"), "continuous.eval.anchors_path")
    if web_cfg.get("cache_dir"):
        _check_data_path(web_cfg.get("cache_dir"), "tools.web.cache_dir")
    if self_patch_cfg.get("queue_dir"):
        _check_data_path(self_patch_cfg.get("queue_dir"), "self_patch.queue_dir")
    if self_patch_cfg.get("sandbox_dir"):
        _check_data_path(self_patch_cfg.get("sandbox_dir"), "self_patch.sandbox_dir")
    if learning.get("raw_path"):
        _check_data_path(learning.get("raw_path"), "learning.raw_path")
    if learning.get("curated_path"):
        _check_data_path(learning.get("curated_path"), "learning.curated_path")
    if learning.get("state_path"):
        _check_data_path(learning.get("state_path"), "learning.state_path")
    if learning.get("evals_path"):
        _check_data_path(learning.get("evals_path"), "learning.evals_path")
    if learning.get("canary_path"):
        _check_data_path(learning.get("canary_path"), "learning.canary_path")

    if missing or errors:
        message = []
        if missing:
            message.append("missing settings keys: " + ", ".join(missing))
        if errors:
            message.append("invalid settings: " + ", ".join(errors))
        raise ValueError("; ".join(message))


def load_settings(profile: str | None = None, settings_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    resolved = resolve_profile(profile)
    if resolved not in profiles:
        raise KeyError(f"Profile '{resolved}' not found in {path}")
    settings = normalize_settings(_resolve_profile(profiles, resolved, []))
    settings["_profile"] = resolved
    settings = _apply_rtx4080_16gb_safe_clamps(settings)
    return settings
