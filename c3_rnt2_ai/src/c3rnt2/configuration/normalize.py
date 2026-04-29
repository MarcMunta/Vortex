from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

from c3rnt2.context_budget import DEFAULT_CONTEXT_BUDGET


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
        runtime.setdefault(
            "paged_lm_head_stream_topk", c3.get("paged_lm_head_stream_topk")
        )
        runtime.setdefault("prefetch_depth", c3.get("prefetch_depth"))
        runtime.setdefault("compression", c3.get("compression"))
        runtime.setdefault("pinned_memory", c3.get("pinned_memory"))
    if "paged_lm_head" not in runtime:
        runtime["paged_lm_head"] = False
    if "cache_vram_budget_mb" not in runtime:
        runtime["cache_vram_budget_mb"] = 2048
    runtime.setdefault("prefetch_depth", 2)
    runtime.setdefault(
        "paged_lm_head_stream_topk",
        runtime.get("paged_lm_head_stream_topk", False) or False,
    )
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
    runtime.setdefault("kv_lowrank_rank", 0)
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
    server_cfg.setdefault("train_host_ram_threshold_mb", 0)
    normalized["server"] = server_cfg

    instructions = normalized.get("instructions", {}) or {}
    instructions.setdefault("vortex_system_path", "config/instructions/vortex_system.md")
    instructions.setdefault("domain_policy_path", "config/instructions/domain_policy.md")
    instructions.setdefault("operator_notes_path", None)
    normalized["instructions"] = instructions

    docker_cfg = normalized.get("docker", {}) or {}
    docker_cfg.setdefault("enabled", False)
    docker_cfg.setdefault("compose_path", "docker-compose.yml")
    docker_cfg.setdefault("runtime_service", "sglang-runtime")
    docker_cfg.setdefault("api_service", "vortex-api")
    docker_cfg.setdefault("trainer_service", "trainer")
    docker_cfg.setdefault("eval_service", "eval")
    normalized["docker"] = docker_cfg

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

    local_lab = normalized.get("local_lab", {}) or {}
    local_lab.setdefault("enabled", False)
    local_lab.setdefault("track", "python_fastapi_react")
    local_lab.setdefault("curriculum_path", "config/local_lab_curriculum.yaml")
    local_lab.setdefault("progress_path", "data/local_lab/progress.json")
    local_lab.setdefault("lessons_path", "data/local_lab/lessons")
    local_lab.setdefault("workspaces_path", "data/local_lab/workspaces")
    local_lab.setdefault("sandbox_root", "data/workspaces")
    local_lab.setdefault("guardrails_enabled", False)
    local_lab.setdefault("lab_confirmation_token", "LAB_CONFIRMED")
    normalized["local_lab"] = local_lab

    voice = normalized.get("voice", {}) or {}
    voice.setdefault("enabled", True)
    voice.setdefault("push_to_talk", True)
    voice.setdefault("vad_enabled", True)
    voice.setdefault("whisper_model", "small")
    voice.setdefault("tts_model", "tts_models/en/ljspeech/tacotron2-DDC")
    voice.setdefault("output_dir", "data/multimodal/voice")
    voice.setdefault("device", "auto")
    voice.setdefault("compute_type", "int8")
    normalized["voice"] = voice

    camera = normalized.get("camera", {}) or {}
    camera.setdefault("enabled", True)
    camera.setdefault("device_id", "default")
    camera.setdefault("frame_width", 960)
    camera.setdefault("frame_height", 540)
    camera.setdefault("fps", 24)
    normalized["camera"] = camera

    gesture = normalized.get("gesture", {}) or {}
    gesture.setdefault("enabled", True)
    gesture.setdefault("mediapipe_enabled", True)
    gesture.setdefault("pinch_threshold", 0.065)
    gesture.setdefault("open_palm_threshold", 0.58)
    gesture.setdefault("fist_threshold", 0.22)
    gesture.setdefault("swipe_velocity_threshold", 0.12)
    gesture.setdefault("dwell_ms", 500)
    gesture.setdefault("debounce_ms", 140)
    gesture.setdefault("smoothing", 0.4)
    gesture.setdefault("model_asset_path", "vortex-chat/public/models/hand_landmarker.task")
    normalized["gesture"] = gesture

    spatial_ui = normalized.get("spatial_ui", {}) or {}
    spatial_ui.setdefault("enabled", True)
    spatial_ui.setdefault("workspace_name", "Spatial Workspace")
    spatial_ui.setdefault("perspective_enabled", True)
    spatial_ui.setdefault("default_perspective", 1100)
    spatial_ui.setdefault("stage_width", 1440)
    spatial_ui.setdefault("stage_height", 900)
    normalized["spatial_ui"] = spatial_ui

    obsidian = normalized.get("obsidian", {}) or {}
    obsidian.setdefault("enabled", True)
    obsidian.setdefault("vault_path", "data/obsidian_vault")
    obsidian.setdefault(
        "folder_map",
        {
            "architecture": "Projects/Vortex/Architecture",
            "session": "Projects/Vortex/Sessions",
            "decision": "Projects/Vortex/Decisions",
            "prompt": "Projects/Vortex/Prompts",
            "bug": "Projects/Vortex/Bugs",
            "experiment": "Projects/Vortex/Experiments",
        },
    )
    normalized["obsidian"] = obsidian

    multimodal_memory = normalized.get("multimodal_memory", {}) or {}
    multimodal_memory.setdefault("enabled", True)
    multimodal_memory.setdefault("state_path", "data/multimodal/spatial_session.json")
    multimodal_memory.setdefault("max_notes", 4)
    multimodal_memory.setdefault("max_chars", 1200)
    normalized["multimodal_memory"] = multimodal_memory

    multimodal_context = normalized.get("multimodal_context", {}) or {}
    multimodal_context.setdefault("enabled", True)
    multimodal_context.setdefault("max_chars", 1800)
    multimodal_context.setdefault("include_memory", True)
    multimodal_context.setdefault("include_spatial_selection", True)
    multimodal_context.setdefault("include_voice", True)
    multimodal_context.setdefault("include_gesture", True)
    normalized["multimodal_context"] = multimodal_context

    context = normalized.get("context", {}) or {}
    context = dict(context) if isinstance(context, dict) else {}
    for key, value in DEFAULT_CONTEXT_BUDGET.items():
        context.setdefault(key, value)
    normalized["context"] = context

    cloud_training = normalized.get("cloud_training", {}) or {}
    cloud_training = dict(cloud_training) if isinstance(cloud_training, dict) else {}
    cloud_training.setdefault("provider", "gcp")
    cloud_training.setdefault("enabled", False)
    cloud_training.setdefault("project_id", None)
    cloud_training.setdefault("region", "us-central1")
    cloud_training.setdefault("bucket", None)
    cloud_training.setdefault("dataset_path", "data/registry/hf_train/qwen_coder_flutter_sft_samples.jsonl")
    cloud_training.setdefault("job_name_prefix", "vortex-qwen-coder")
    cloud_training.setdefault("service_account_env", "GOOGLE_APPLICATION_CREDENTIALS")
    normalized["cloud_training"] = cloud_training

    presentation = normalized.get("presentation", {}) or {}
    presentation.setdefault("default_panel_type", "presentation")
    presentation.setdefault("page_step", 1)
    presentation.setdefault("swipe_enabled", True)
    normalized["presentation"] = presentation

    workspace_panels = normalized.get("workspace_panels", {}) or {}
    workspace_panels.setdefault("max_active_panels", 12)
    workspace_panels.setdefault("default_width", 360)
    workspace_panels.setdefault("default_height", 220)
    workspace_panels.setdefault(
        "default_kinds",
        ["note", "presentation", "browser", "image", "obsidian", "sketch"],
    )
    normalized["workspace_panels"] = workspace_panels

    adapters = normalized.get("adapters", {}) or {}
    adapters.setdefault("enabled", False)
    adapters.setdefault("allow_empty", False)
    adapters.setdefault("paths", {})
    adapters.setdefault("max_loaded", 0)
    adapters.setdefault("default", None)
    adapter_router = adapters.get("router", {}) or {}
    adapter_router.setdefault("mode", "keyword_map")
    adapter_router.setdefault("keyword_map", {})
    adapter_router.setdefault("default", adapters.get("default"))
    adapter_router.setdefault(
        "embedding_backend", knowledge.get("embedding_backend", "hash")
    )
    adapter_router.setdefault("embedding_dim", 128)
    adapter_router.setdefault("embedding_min_score", 0.0)
    adapter_router.setdefault("top_k", 1)
    adapter_router.setdefault("mix_mode", "single")
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
    hf_train.setdefault("gradient_checkpointing", True)
    hf_train.setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    hf_train.setdefault("extra_training_paths", [])
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
    if cont.get("run_interval_minutes") is not None:
        warnings.warn(
            "continuous.run_interval_minutes is deprecated; use continuous.interval_minutes",
            DeprecationWarning,
        )
        cont.setdefault("interval_minutes", cont.get("run_interval_minutes"))
    if (
        "run_interval_minutes" not in cont
        and cont.get("interval_minutes") is not None
    ):
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
    local_sources = cont.get("local_sources", {}) or {}
    local_sources.setdefault("enabled", False)
    local_sources.setdefault("include_repo", False)
    local_sources.setdefault("include_local_corpus", False)
    local_sources.setdefault("include_lessons", False)
    local_sources.setdefault("include_logs", True)
    local_sources.setdefault("include_memory", True)
    local_sources.setdefault(
        "repo_paths",
        ["README.md", "src", "tests", "scripts", "docs", "config"],
    )
    local_sources.setdefault("corpus_paths", ["data/corpora/programming"])
    local_sources.setdefault("lesson_paths", ["data/local_lab/lessons"])
    local_sources.setdefault(
        "include_globs",
        [
            "*.py",
            "*.pyi",
            "*.md",
            "*.txt",
            "*.json",
            "*.jsonl",
            "*.yaml",
            "*.yml",
            "*.toml",
            "*.ts",
            "*.tsx",
            "*.js",
            "*.jsx",
            "*.ps1",
            "*.sh",
        ],
    )
    local_sources.setdefault(
        "exclude_globs",
        [
            "**/__pycache__/**",
            "**/.git/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "data/web_cache/**",
            "data/logs/**",
            "data/checkpoints/**",
            "data/models/**",
            "data/hf_offload/**",
            "*.pt",
            "*.bin",
            "*.safetensors",
            "*.gguf",
        ],
    )
    cont["local_sources"] = local_sources
    normalized["continuous"] = cont

    autopilot = normalized.get("autopilot", {}) or {}
    autopilot.setdefault("enabled", False)
    autopilot.setdefault(
        "interval_minutes", cont.get("interval_minutes", 30) if cont else 30
    )
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
    autopilot.setdefault("autopatch_require_approval", True)
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
    autopilot.setdefault(
        "todo_regex", r"TODO\((P1|PRIORITY)\)|TODO!|TODO:HIGH|TODO:CRITICAL"
    )
    normalized["autopilot"] = autopilot

    if lava:
        normalized["lava"] = lava

    profile_contract = normalized.get("profile_contract", {}) or {}
    profile_contract.setdefault("offline_required", False)
    profile_contract.setdefault("require_web_disabled", False)
    profile_contract.setdefault("require_ollama", False)
    profile_contract.setdefault("require_external_engine", None)
    profile_contract.setdefault("require_local_base_url", False)
    profile_contract.setdefault("require_wsl_training", False)
    profile_contract.setdefault("require_docker", False)
    profile_contract.setdefault("disable_fallbacks", False)
    profile_contract.setdefault("approved_training_sources_only", False)
    profile_contract.setdefault("min_host_ram_free_mb", 0)
    normalized["profile_contract"] = profile_contract

    return normalized
