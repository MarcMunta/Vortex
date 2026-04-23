from __future__ import annotations

import os
from pathlib import Path

from .contracts import _validate_profile_contract


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
    local_sources_cfg = cont.get("local_sources", {}) or {}
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    self_patch_cfg = settings.get("self_patch", {}) or {}
    hf_train_cfg = settings.get("hf_train", {}) or {}
    adapters_cfg = settings.get("adapters", {}) or {}
    server_cfg = settings.get("server", {}) or {}
    voice_cfg = settings.get("voice", {}) or {}
    camera_cfg = settings.get("camera", {}) or {}
    gesture_cfg = settings.get("gesture", {}) or {}
    spatial_ui_cfg = settings.get("spatial_ui", {}) or {}
    obsidian_cfg = settings.get("obsidian", {}) or {}
    multimodal_memory_cfg = settings.get("multimodal_memory", {}) or {}
    multimodal_context_cfg = settings.get("multimodal_context", {}) or {}
    presentation_cfg = settings.get("presentation", {}) or {}
    workspace_panels_cfg = settings.get("workspace_panels", {}) or {}

    if not tok.get("vortex_tok_path"):
        missing.append("tokenizer.vortex_tok_path")
    if backend == "hf":
        if not core.get("hf_model"):
            missing.append("core.hf_model")
    elif backend == "llama_cpp":
        if not core.get("llama_cpp_model_path"):
            missing.append("core.llama_cpp_model_path")
    elif backend == "tensorrt":
        if not (core.get("tensorrt_engine_dir") or core.get("tensorrt_engine_path")):
            missing.append("core.tensorrt_engine_dir")
        if not (core.get("tensorrt_tokenizer") or core.get("hf_model")):
            missing.append("core.tensorrt_tokenizer or core.hf_model")
    elif backend in {"external", "vllm", "sglang"}:
        engine = (
            str(core.get("external_engine") or core.get("engine") or backend)
            .strip()
            .lower()
        )
        if backend == "external" and engine not in {"vllm", "sglang", "ollama", "lmstudio"}:
            errors.append(
                "core.external_engine must be vllm, sglang, ollama, or lmstudio when core.backend=external"
            )
        base_url = core.get("external_base_url") or core.get("external_url")
        if not base_url:
            missing.append("core.external_base_url")
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
    if kv_quant in {"low_rank", "low-rank", "mla"}:
        kv_quant = "lowrank"
    if kv_quant not in {"none", "int8", "2bit", "lowrank"}:
        errors.append("runtime.kv_quant must be one of none|int8|2bit|lowrank")
    if kv_quant == "2bit":
        if not bool(runtime.get("kv_quant_2bit_experimental", False)):
            errors.append(
                "runtime.kv_quant=2bit is experimental; set runtime.kv_quant_2bit_experimental=true"
            )
        if not bool(runtime.get("i_know_what_im_doing", False)):
            errors.append(
                "runtime.kv_quant=2bit requires runtime.i_know_what_im_doing=true"
            )
    if kv_quant == "lowrank":
        raw_rank = runtime.get("kv_lowrank_rank")
        try:
            rank_i = int(raw_rank) if raw_rank is not None else 0
        except Exception:
            rank_i = None
        if rank_i is None:
            errors.append("runtime.kv_lowrank_rank must be an integer (0 = auto)")
        elif rank_i < 0:
            errors.append("runtime.kv_lowrank_rank must be >= 0 (0 = auto)")
        elif rank_i > 0 and backend not in {"hf", "llama_cpp", "tensorrt"}:
            try:
                hidden = int(core.get("hidden_size", 0) or 0)
            except Exception:
                hidden = 0
            if hidden > 0 and rank_i >= hidden:
                errors.append("runtime.kv_lowrank_rank must be < core.hidden_size")
    gpu_decompress = str(runtime.get("gpu_decompress", "none")).lower()
    if gpu_decompress not in {"none", "triton"}:
        errors.append(
            "runtime.gpu_decompress must be none or triton (CPU decompress + H2D pipeline)"
        )

    if web_cfg:
        if bool(web_cfg.get("enabled", False)) and not web_cfg.get("allow_domains"):
            errors.append(
                "tools.web.allow_domains required when tools.web.enabled is true"
            )
        if bool(web_cfg.get("enabled", False)) and not web_cfg.get(
            "allow_content_types"
        ):
            errors.append(
                "tools.web.allow_content_types required when tools.web.enabled is true"
            )
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
            errors.append(
                "hf_train.model_name or core.hf_model required for hf training"
            )
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

    if adapters_cfg and bool(adapters_cfg.get("enabled", False)):
        paths = adapters_cfg.get("paths", {}) or {}
        allow_empty = bool(adapters_cfg.get("allow_empty", False))
        if not paths and not allow_empty:
            errors.append(
                "adapters.paths must not be empty when adapters.enabled is true"
            )
        router_cfg = adapters_cfg.get("router", {}) or {}
        keyword_map = router_cfg.get("keyword_map", {}) or {}
        try:
            top_k = int(router_cfg.get("top_k", 1) or 1)
            if top_k <= 0:
                errors.append("adapters.router.top_k must be >= 1")
        except Exception:
            errors.append("adapters.router.top_k must be >= 1")
        mix_mode = str(router_cfg.get("mix_mode", "single") or "single").lower()
        if mix_mode not in {"single", "weighted"}:
            errors.append("adapters.router.mix_mode must be one of single|weighted")
        for _kw, name in keyword_map.items():
            if name and name not in paths:
                errors.append(
                    f"adapters.router.keyword_map references unknown adapter: {name}"
                )
        default = adapters_cfg.get("default") or router_cfg.get("default")
        if default and default not in paths:
            errors.append(f"adapters.default unknown: {default}")
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
    train_host_ram_threshold_mb = server_cfg.get("train_host_ram_threshold_mb")
    if (
        train_host_ram_threshold_mb is not None
        and int(train_host_ram_threshold_mb) < 0
    ):
        errors.append("server.train_host_ram_threshold_mb must be >= 0")

    if voice_cfg:
        if not str(voice_cfg.get("whisper_model") or "").strip():
            errors.append("voice.whisper_model must not be empty")
        if not str(voice_cfg.get("tts_model") or "").strip():
            errors.append("voice.tts_model must not be empty")
    for key in ("frame_width", "frame_height", "fps"):
        raw_value = camera_cfg.get(key)
        if raw_value is not None and int(raw_value) <= 0:
            errors.append(f"camera.{key} must be > 0")
    for key in ("pinch_threshold", "open_palm_threshold", "fist_threshold", "swipe_velocity_threshold", "smoothing"):
        raw_value = gesture_cfg.get(key)
        if raw_value is not None and float(raw_value) < 0:
            errors.append(f"gesture.{key} must be >= 0")
    for key in ("dwell_ms", "debounce_ms"):
        raw_value = gesture_cfg.get(key)
        if raw_value is not None and int(raw_value) < 0:
            errors.append(f"gesture.{key} must be >= 0")
    for key in ("default_perspective", "stage_width", "stage_height"):
        raw_value = spatial_ui_cfg.get(key)
        if raw_value is not None and float(raw_value) <= 0:
            errors.append(f"spatial_ui.{key} must be > 0")
    if obsidian_cfg and not isinstance(obsidian_cfg.get("folder_map", {}), dict):
        errors.append("obsidian.folder_map must be a mapping")
    if multimodal_memory_cfg.get("max_notes") is not None and int(multimodal_memory_cfg.get("max_notes", 1)) <= 0:
        errors.append("multimodal_memory.max_notes must be > 0")
    if multimodal_memory_cfg.get("max_chars") is not None and int(multimodal_memory_cfg.get("max_chars", 1)) <= 0:
        errors.append("multimodal_memory.max_chars must be > 0")
    if multimodal_context_cfg.get("max_chars") is not None and int(multimodal_context_cfg.get("max_chars", 1)) <= 0:
        errors.append("multimodal_context.max_chars must be > 0")
    if presentation_cfg.get("page_step") is not None and int(presentation_cfg.get("page_step", 1)) <= 0:
        errors.append("presentation.page_step must be > 0")
    if workspace_panels_cfg.get("max_active_panels") is not None and int(workspace_panels_cfg.get("max_active_panels", 1)) <= 0:
        errors.append("workspace_panels.max_active_panels must be > 0")
    for key in ("default_width", "default_height"):
        raw_value = workspace_panels_cfg.get(key)
        if raw_value is not None and float(raw_value) <= 0:
            errors.append(f"workspace_panels.{key} must be > 0")
    if not isinstance(workspace_panels_cfg.get("default_kinds", []), list):
        errors.append("workspace_panels.default_kinds must be a list")

    if bool(local_sources_cfg.get("enabled", False)):
        for key in ("repo_paths", "corpus_paths", "lesson_paths"):
            raw_paths = local_sources_cfg.get(key, [])
            if not isinstance(raw_paths, list):
                errors.append(f"continuous.local_sources.{key} must be a list")
        for key in ("include_globs", "exclude_globs"):
            raw_globs = local_sources_cfg.get(key, [])
            if not isinstance(raw_globs, list):
                errors.append(f"continuous.local_sources.{key} must be a list")

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

    def _check_writable_path(path_value: str | Path | None, label: str) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"{label} parent not writable: {exc}")
            return
        if not os.access(path.parent, os.W_OK):
            errors.append(f"{label} parent not writable")

    _check_data_path(cont.get("knowledge_path"), "continuous.knowledge_path")
    _check_data_path(cont.get("replay", {}).get("path"), "continuous.replay.path")
    _check_data_path(
        cont.get("eval", {}).get("anchors_path"), "continuous.eval.anchors_path"
    )
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
    local_lab = settings.get("local_lab", {}) or {}
    if local_lab.get("progress_path"):
        _check_writable_path(local_lab.get("progress_path"), "local_lab.progress_path")
    if local_lab.get("lessons_path"):
        _check_writable_path(local_lab.get("lessons_path"), "local_lab.lessons_path")
    if local_lab.get("workspaces_path"):
        _check_writable_path(local_lab.get("workspaces_path"), "local_lab.workspaces_path")
    if local_lab.get("sandbox_root"):
        _check_data_path(local_lab.get("sandbox_root"), "local_lab.sandbox_root")
    if voice_cfg.get("output_dir"):
        _check_writable_path(voice_cfg.get("output_dir"), "voice.output_dir")
    if multimodal_memory_cfg.get("state_path"):
        _check_data_path(multimodal_memory_cfg.get("state_path"), "multimodal_memory.state_path")
    if obsidian_cfg.get("vault_path"):
        _check_writable_path(obsidian_cfg.get("vault_path"), "obsidian.vault_path")
    instructions_cfg = settings.get("instructions", {}) or {}
    def _check_existing_text_path(path_value: str | Path | None, label: str) -> None:
        if not path_value:
            return
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            errors.append(f"{label} missing")
            return
        if not path.is_file():
            errors.append(f"{label} must be a file")
    _check_existing_text_path(
        instructions_cfg.get("vortex_system_path"), "instructions.vortex_system_path"
    )
    _check_existing_text_path(
        instructions_cfg.get("domain_policy_path"), "instructions.domain_policy_path"
    )
    _check_existing_text_path(
        instructions_cfg.get("operator_notes_path"), "instructions.operator_notes_path"
    )
    docker_cfg = settings.get("docker", {}) or {}
    if docker_cfg.get("compose_path"):
        compose_path = Path(docker_cfg.get("compose_path"))
        if not compose_path.is_absolute():
            compose_path = (base_dir / compose_path).resolve()
        if bool(docker_cfg.get("enabled", False)) and not compose_path.exists():
            errors.append("docker.compose_path missing")
    if hf_train_cfg.get("registry_dir"):
        _check_data_path(hf_train_cfg.get("registry_dir"), "hf_train.registry_dir")
    if hf_train_cfg.get("dataset_path"):
        _check_data_path(hf_train_cfg.get("dataset_path"), "hf_train.dataset_path")
    if hf_train_cfg.get("state_path"):
        _check_data_path(hf_train_cfg.get("state_path"), "hf_train.state_path")
    for idx, extra_path in enumerate(hf_train_cfg.get("extra_training_paths", []) or []):
        _check_writable_path(extra_path, f"hf_train.extra_training_paths[{idx}]")

    _validate_profile_contract(settings, errors)

    if missing or errors:
        message = []
        if missing:
            message.append("missing settings keys: " + ", ".join(missing))
        if errors:
            message.append("invalid settings: " + ", ".join(errors))
        raise ValueError("; ".join(message))
