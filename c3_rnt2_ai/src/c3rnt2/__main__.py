from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from types import SimpleNamespace
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .config import load_settings, resolve_profile, validate_profile, resolve_web_allowlist, resolve_web_strict
from .continuous.dataset import ingest_sources, collect_samples
from .continuous.bootstrap import run_bootstrap
from .device import detect_device
from .doctor import check_deps, run_deep_checks
from .model_loader import load_inference_model
from .prompting.chat_format import build_chat_prompt
from .server import run_server
from .utils.locks import LockUnavailable, acquire_exclusive_lock, FileLock, is_lock_held
from .agent.agent_loop import run_demo_agent
from .training.train_backend import train_once_backend
from .utils.oom import is_oom_error, clear_cuda_cache
from .learning_loop.data_collector import collect_from_episodes
from .learning_loop.data_curator import curate_dataset
from .learning_loop.trainer import train_qlora
from .learning_loop.evaluator import evaluate_adapter, log_eval
from .learning_loop.promoter import promote_latest
from .agent.runner import run_agent
from .runtime.vram_governor import decide_max_new_tokens
from .autopilot import run_autopilot_loop, run_autopilot_tick, run_autopatch_once
from .continuous.promotion import promote_quarantine_run


def _load_and_validate(profile: str | None, override: Callable[[dict], dict] | None = None) -> dict:
    settings = load_settings(profile)
    if override is not None:
        settings = override(settings)
    validate_profile(settings, base_dir=Path("."))
    return settings


def _resolve_allowlist(settings: dict) -> list[str]:
    return resolve_web_allowlist(settings)


def _supported_agent_tools() -> set[str]:
    return {
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
    }


def _strict_web_ingest(settings: dict) -> bool:
    cont = settings.get("continuous", {}) or {}
    strict_flag = cont.get("strict_web_ingest")
    if strict_flag is not None:
        return bool(strict_flag)
    if bool(settings.get("hf_train", {}).get("enabled", False)):
        return True
    profile_name = settings.get("_profile")
    return bool(profile_name == "safe_selftrain_4080")


def _check_web_ingest(settings: dict, allowlist: list[str]) -> tuple[bool, str | None, bool]:
    cont = settings.get("continuous", {}) or {}
    ingest_web = bool(cont.get("ingest_web", True))
    tools_web = settings.get("tools", {}).get("web", {}) or {}
    web_enabled = bool(tools_web.get("enabled", False))
    if ingest_web and allowlist and not web_enabled:
        return False, "ingest_web enabled but tools.web.enabled=false", _strict_web_ingest(settings)
    return True, None, False


def _coerce_train_result(payload: dict | object) -> SimpleNamespace:
    if isinstance(payload, SimpleNamespace):
        return payload
    if isinstance(payload, dict):
        return SimpleNamespace(**payload)
    if hasattr(payload, "__dict__"):
        try:
            return SimpleNamespace(**dict(getattr(payload, "__dict__", {})))
        except Exception:
            return SimpleNamespace(ok=False, ok_eval=False, ok_train=False, error="invalid_train_result")
    return SimpleNamespace(ok=False, ok_eval=False, ok_train=False, error="invalid_train_result")


def _run_train_subprocess(settings: dict, reuse_dataset: bool, *, timeout_s: float | None = None) -> dict:
    profile = settings.get("_profile") or resolve_profile(None)
    cmd = [sys.executable, "-m", "c3rnt2", "train-once", "--profile", str(profile)]
    if reuse_dataset:
        cmd.append("--reuse-dataset")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "ok_train": False, "ok_eval": False, "error": "train_subprocess_timeout"}
    if result.returncode != 0:
        return {"ok": False, "ok_train": False, "ok_eval": False, "error": result.stderr.strip() or "train subprocess failed"}
    payload = None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if payload is None:
        return {"ok": False, "ok_train": False, "ok_eval": False, "error": "train subprocess output not parseable"}
    return payload


def _run_train_inprocess(
    settings: dict,
    base_dir: Path,
    reuse_dataset: bool,
    *,
    max_steps: int | None = None,
    train_fn: Callable | None = None,
) -> SimpleNamespace:
    if train_fn is not None:
        result = train_fn(settings, base_dir, reuse_dataset=reuse_dataset)
    else:
        result = train_once_backend(settings, base_dir, reuse_dataset=reuse_dataset, max_steps=max_steps)
    return _coerce_train_result(result)


def _unload_models_for_train(app: object) -> None:
    state = getattr(app, "state", None)
    if state is None:
        return
    lock = getattr(state, "model_lock", None)
    ctx = lock.write_lock() if lock and hasattr(lock, "write_lock") else nullcontext()
    with ctx:
        try:
            state.model = None
            state.models = {}
        except Exception:
            pass
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _reload_models_for_app(app: object, settings: dict) -> None:
    state = getattr(app, "state", None)
    if state is None:
        return
    lock = getattr(state, "model_lock", None)
    ctx = lock.write_lock() if lock and hasattr(lock, "write_lock") else nullcontext()
    with ctx:
        model = load_inference_model(settings)
        backend = str(settings.get("core", {}).get("backend", "vortex")).lower()
        label = "hf" if backend == "hf" else "core"
        state.model = model
        state.models = {label: model}


def _self_patch_queue_dir(base_dir: Path, settings: dict) -> Path:
    queue_dir = settings.get("self_patch", {}).get("queue_dir", "data/self_patch/queue")
    qpath = Path(queue_dir)
    if not qpath.is_absolute():
        qpath = base_dir / qpath
    return qpath


def _apply_cli_overrides(settings: dict, args: argparse.Namespace) -> dict:
    core = dict(settings.get("core", {}) or {})
    if getattr(args, "backend", None):
        core["backend"] = str(args.backend)
    if getattr(args, "model", None):
        core["hf_model"] = str(args.model)
    if getattr(args, "device", None):
        core["hf_device"] = str(args.device)
    backend = str(core.get("backend", "vortex")).lower()
    if backend == "hf":
        info = detect_device()
        core.setdefault("hf_device", info.device if info.cuda_available else "cpu")
        core.setdefault("dtype", info.dtype)
        if info.cuda_available and info.vram_gb and "hf_load_in_4bit" not in core:
            if info.vram_gb <= 16:
                core["hf_load_in_4bit"] = True
    settings["core"] = core
    return settings


def _apply_serve_self_train_defaults(settings: dict) -> dict:
    server_cfg = dict(settings.get("server", {}) or {})
    server_cfg.setdefault("block_during_training", True)
    settings["server"] = server_cfg
    return settings


def _resolve_fallback_backend(settings: dict, current: str) -> str | None:
    core = settings.get("core", {}) or {}
    fallback = core.get("backend_fallback") or core.get("hf_fallback")
    if fallback:
        fb = str(fallback).lower()
        if fb != current:
            return fb
    if current != "hf" and core.get("hf_model"):
        return "hf"
    return None


def _resolve_interval_minutes(args_interval: float | None, settings: dict) -> float:
    if args_interval is not None:
        return float(args_interval)
    cont = settings.get("continuous", {}) or {}
    if cont.get("run_interval_minutes") is not None:
        return float(cont.get("run_interval_minutes"))
    if cont.get("interval_minutes") is not None:
        return float(cont.get("interval_minutes"))
    return 30.0


def _run_module(module: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + extra_args
    subprocess.run(cmd, check=True)


def _check_path_writable(base_dir: Path, path_value: str | Path | None) -> tuple[bool, str]:
    if not path_value:
        return True, ""
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    target = path
    is_file = path.suffix != ""
    check = path.parent if is_file else path
    while not check.exists() and check.parent != check:
        check = check.parent
    if not check.exists():
        return False, f"{target} parent missing"
    if not os.access(check, os.W_OK):
        return False, f"{target} parent not writable"
    return True, ""


def _run_doctor_checks(settings: dict, base_dir: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, object] = {}

    profile_name = settings.get("_profile")
    tools_cfg = settings.get("tools", {}) or {}
    if profile_name == "safe_selftrain_4080_hf" and isinstance(tools_cfg, dict) and "autopilot" in tools_cfg:
        errors.append("autopilot config must be top-level (autopilot:), not tools.autopilot")

    supported = _supported_agent_tools()
    tools_enabled = settings.get("agent", {}).get("tools_enabled")
    if tools_enabled is None:
        tools_enabled = list(supported)
    bad_tools = [str(t) for t in tools_enabled if str(t) not in supported]
    if bad_tools:
        errors.append(f"unsupported tools_enabled: {bad_tools}")
    info["tools_enabled"] = list(tools_enabled)

    allowlist = _resolve_allowlist(settings)
    cont_cfg = settings.get("continuous", {}) or {}
    if bool(cont_cfg.get("ingest_web", True)) and not allowlist:
        return {"ok": False, "error": "ingest_web enabled but allow_domains empty", "skipped": "web_ingest_unavailable"}
    web_ok, web_msg, strict = _check_web_ingest(settings, allowlist)
    if not web_ok:
        if strict:
            errors.append(web_msg or "web_ingest_invalid")
        else:
            warnings.append(web_msg or "web_ingest_invalid")
    info["web_ingest"] = {"ok": web_ok, "strict": strict, "allowlist": allowlist}
    cont = settings.get("continuous", {}) or {}
    if bool(cont.get("ingest_web", True)) and not allowlist:
        errors.append("ingest_web enabled but allow_domains is empty")

    tools_web = settings.get("tools", {}).get("web", {}) or {}
    hf_train = settings.get("hf_train", {}) or {}
    paths = []
    if cont.get("knowledge_path"):
        paths.append(("continuous.knowledge_path", cont.get("knowledge_path")))
    if tools_web.get("cache_dir"):
        paths.append(("tools.web.cache_dir", tools_web.get("cache_dir")))
    if hf_train.get("registry_dir"):
        paths.append(("hf_train.registry_dir", hf_train.get("registry_dir")))
    for label, value in paths:
        ok, msg = _check_path_writable(base_dir, value)
        if not ok:
            errors.append(f"{label}: {msg}")

    backend = str((settings.get("core", {}) or {}).get("backend", "vortex")).lower()
    try:
        # Dry checks only: avoid pulling large HF weights during "doctor".
        if backend == "hf":
            __import__("transformers")
        elif backend == "tensorrt":
            __import__("tensorrt")
        else:
            __import__("torch")
        info["model_load_ok"] = True
        info["model_load_mode"] = "dry"
    except Exception as exc:
        errors.append(f"model_load_failed: {exc}")

    # Windows + CUDA gate for GPU-targeted profiles (e.g. RTX 4080).
    # Only enforce when the profile/device indicates CUDA is expected.
    profile_l = str(settings.get("_profile") or "").lower()
    core = settings.get("core", {}) or {}
    requested_device = core.get("hf_device", core.get("device"))
    requested_device_l = str(requested_device or "").lower()
    wants_cuda = ("rtx4080" in profile_l or "4080" in profile_l) and not (requested_device_l.startswith("cpu") or requested_device_l in {"", "none"})
    if sys.platform.startswith("win") and wants_cuda:
        try:
            import torch  # type: ignore

            if not bool(torch.cuda.is_available()):
                errors.append(
                    "cuda_missing: Windows profile expects CUDA but torch.cuda.is_available() is False. "
                    "Install a CUDA-enabled PyTorch build and NVIDIA drivers, or use WSL2 for training."
                )
        except Exception as exc:
            errors.append(f"cuda_check_failed: {exc}")

    # HF training dependency gate (do not block inference; fail doctor if training is enabled but stack is missing).
    hf_train = settings.get("hf_train", {}) or {}
    if bool(hf_train.get("enabled", False)):
        missing: list[str] = []
        for mod in ("transformers", "peft", "accelerate"):
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)
        quant_requested = bool(
            hf_train.get("load_in_4bit")
            or hf_train.get("load_in_8bit")
            or core.get("hf_load_in_4bit")
            or core.get("hf_load_in_8bit")
        )
        if quant_requested:
            try:
                __import__("bitsandbytes")
            except Exception:
                missing.append("bitsandbytes")
        if missing:
            errors.append(
                "hf_train_deps_missing: "
                + ", ".join(sorted(set(missing)))
                + ". Recommendation: on Windows, use WSL2 for HF training (PEFT/bitsandbytes) or install the missing packages."
            )

    decode_cfg = settings.get("decode", {}) or {}
    requested = int(decode_cfg.get("max_new_tokens", 64))
    device = core.get("hf_device", core.get("device", None))
    dtype = core.get("dtype")
    decided = decide_max_new_tokens(requested, device, dtype, settings)
    info["vram_governor"] = {"requested": requested, "decided": decided, "device": device, "dtype": dtype}

    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}


def _run_doctor_deep_checks(settings: dict, base_dir: Path) -> dict:
    errors: list[str] = []
    info: dict[str, object] = {}
    deep_settings = json.loads(json.dumps(settings))
    cont = deep_settings.setdefault("continuous", {})
    cont["ingest_web"] = False
    cont.setdefault("trigger", {})["enabled"] = False
    server_cfg = deep_settings.setdefault("server", {})
    server_cfg["train_strategy"] = "inprocess"
    app = SimpleNamespace(state=SimpleNamespace())
    dummy_result = SimpleNamespace(ok=True, ok_train=True, ok_eval=True, loss=0.0, steps=1, samples=1, tokens_per_sec=1.0, vram_peak_mb=None)
    res = _run_self_train_tick(
        app,
        deep_settings,
        base_dir,
        reuse_dataset=True,
        maintenance_window_s=0.0,
        reload_fn=None,
        train_fn=lambda *_args, **_kwargs: dummy_result,
    )
    info["self_train_mock_tick"] = res
    if not res.get("ok", False):
        errors.append("self_train_tick_failed")

    auto_res = run_autopilot_tick(deep_settings, base_dir, no_web=True, mock=True, force=False)
    info["autopilot_mock_tick"] = {"ok": auto_res.ok, "steps": auto_res.steps, "error": auto_res.error}
    if not auto_res.ok and not (auto_res.error and "lock" in auto_res.error):
        errors.append("autopilot_tick_failed")
    ap_cfg = deep_settings.get("autopilot", {}) or {}
    if not bool(ap_cfg.get("enabled", False)) and auto_res.ok and isinstance(auto_res.steps, dict):
        if auto_res.steps.get("skipped") != "disabled":
            errors.append("autopilot_disabled_not_respected")

    try:
        # Exercise serve-autopilot loop logic (mock) without spinning up uvicorn.
        ap_settings = json.loads(json.dumps(deep_settings))
        ap_settings.setdefault("autopilot", {})["enabled"] = True
        ap_settings["autopilot"]["train_cooldown_minutes"] = 0

        def _dummy_train(_profile: str, reuse_dataset: bool, max_steps: int | None):
            _ = _profile, reuse_dataset, max_steps
            return {"ok": False, "ok_train": False, "ok_eval": False, "error": "dummy_train"}

        tick = run_autopilot_tick(ap_settings, base_dir, no_web=True, mock=False, force=True, train_runner=_dummy_train)
        info["serve_autopilot_mock_once"] = {"ok": tick.ok, "steps": tick.steps, "error": tick.error}
    except Exception as exc:
        errors.append(f"serve_autopilot_mock_failed: {exc}")

    strategy = str((settings.get("server", {}) or {}).get("train_strategy", "subprocess")).lower()
    if strategy == "subprocess":
        try:
            __import__("c3rnt2.__main__")
            info["train_strategy_check"] = "subprocess_import_ok"
        except Exception as exc:
            errors.append(f"train_strategy_subprocess_import_failed: {exc}")
    else:
        info["train_strategy_check"] = f"{strategy}_ok"

    if is_lock_held(base_dir, "serve") and strategy in {"subprocess", "subprocess_unload"}:
        info["train_strategy_warning"] = "server_lock_held + subprocess strategy may OOM (prefer unload_reload/subprocess_unload)"

    return {"ok": not errors, "errors": errors, "info": info}


def cmd_doctor(args: argparse.Namespace) -> None:
    info = detect_device()
    device_payload = {
        "device": info.device,
        "cuda_available": info.cuda_available,
        "gpu": info.name,
        "vram_gb": info.vram_gb,
        "dtype": info.dtype,
        "python": sys.version.split()[0],
    }
    print(device_payload)
    modules = [
        "torch",
        "bitsandbytes",
        "faiss",
        "triton",
        "fastapi",
        "zstandard",
        "lz4",
    ]
    status = check_deps(modules)
    print({"deps": status})

    base_dir = Path(".")
    try:
        settings = _load_and_validate(args.profile)
        print({"settings_ok": True, "profile": resolve_profile(args.profile)})
    except Exception as exc:
        print({"warning": "settings_invalid", "error": str(exc)})
        sys.exit(1)

    base_report = _run_doctor_checks(settings, base_dir)
    print(base_report)
    if not base_report.get("ok", False):
        final = {
            "ok": False,
            "profile": resolve_profile(args.profile),
            "device": device_payload,
            "deps": status,
            "base": base_report,
        }
        print(json.dumps(final, ensure_ascii=True))
        sys.exit(1)

    if args.deep:
        deep_result: dict | None = None
        try:
            deep_result = run_deep_checks(settings, base_dir=base_dir, mock=bool(getattr(args, "mock", False)))
        except Exception as exc:
            deep_result = {"deep_ok": False, "error": str(exc)}
        final = {
            "ok": bool(deep_result is not None and deep_result.get("deep_ok", False)),
            "profile": resolve_profile(args.profile),
            "device": device_payload,
            "deps": status,
            "base": base_report,
            "deep": deep_result,
        }
        out_path = base_dir / "data" / "doctor" / "last.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"doctor --deep {'OK' if final['ok'] else 'FAIL'} (wrote {out_path})")
        print(json.dumps(final, ensure_ascii=True))
        if not final["ok"]:
            sys.exit(1)
        return

    final = {
        "ok": True,
        "profile": resolve_profile(args.profile),
        "device": device_payload,
        "deps": status,
        "base": base_report,
    }
    print(json.dumps(final, ensure_ascii=True))


def cmd_chat(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile, override=lambda s: _apply_cli_overrides(s, args))
    model = load_inference_model(settings)
    info = detect_device()
    print({"device": info.device, "vram_gb": info.vram_gb, "dtype": info.dtype})
    decode_cfg = settings.get("decode", {}) or {}
    default_system = settings.get("core", {}).get("hf_system_prompt", "You are a helpful coding assistant.")
    backend = settings.get("core", {}).get("backend", "vortex")
    print("VORTEX-X chat. Type 'exit' to quit.")
    while True:
        prompt = input("> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        messages = [{"role": "user", "content": prompt}]
        prompt_text = build_chat_prompt(messages, backend, tokenizer=getattr(model, "tokenizer", None), default_system=default_system)
        max_new = args.max_new_tokens or int(decode_cfg.get("max_new_tokens", 64))
        device = getattr(model, "device", None) or settings.get("core", {}).get("hf_device")
        dtype = getattr(model, "dtype", None) or settings.get("core", {}).get("dtype")
        max_new = decide_max_new_tokens(max_new, device, dtype, settings)
        temperature = args.temperature if args.temperature is not None else float(decode_cfg.get("temperature", 1.0))
        top_p = args.top_p if args.top_p is not None else float(decode_cfg.get("top_p", 1.0))
        repetition_penalty = float(decode_cfg.get("repetition_penalty", 1.0))
        no_repeat_ngram = int(decode_cfg.get("no_repeat_ngram", 0))
        penalty_window = int(decode_cfg.get("penalty_window", 512))
        top_p_min_k = int(decode_cfg.get("top_p_min_k", 128))
        top_p_max_k = int(decode_cfg.get("top_p_max_k", 512))
        if args.stream:
            for attempt in range(2):
                try:
                    if hasattr(model, "stream_generate"):
                        for chunk in model.stream_generate(prompt_text, max_new_tokens=max_new, temperature=temperature, top_p=top_p):
                            print(chunk, end="", flush=True)
                    else:
                        from .server import _stream_generate

                        for chunk in _stream_generate(
                            model,
                            prompt_text,
                            max_new,
                            temperature,
                            top_p,
                            repetition_penalty,
                            no_repeat_ngram,
                            penalty_window,
                            top_p_min_k,
                            top_p_max_k,
                        ):
                            print(chunk, end="", flush=True)
                    print()
                    break
                except RuntimeError as exc:
                    if is_oom_error(exc) and attempt == 0:
                        clear_cuda_cache()
                        fb = _resolve_fallback_backend(settings, str(backend).lower())
                        if fb:
                            model = load_inference_model(settings, backend_override=fb)
                            backend = fb
                            continue
                    raise
            continue
        try:
            response = model.generate(
                prompt_text,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
            )
            print(response)
        except RuntimeError as exc:
            if is_oom_error(exc):
                clear_cuda_cache()
                fb = _resolve_fallback_backend(settings, str(backend).lower())
                if fb:
                    model = load_inference_model(settings, backend_override=fb)
                    backend = fb
                    response = model.generate(
                        prompt_text,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    print(response)
                    continue
            raise


def cmd_serve(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile, override=lambda s: _apply_cli_overrides(s, args))
    base_dir = Path(".")
    try:
        lock = acquire_exclusive_lock(base_dir, "serve")
    except LockUnavailable:
        print({"ok": False, "error": "serve lock unavailable (train/self_patch running?)"})
        return
    try:
        run_server(settings, base_dir=base_dir, host=args.host, port=args.port)
    finally:
        lock.release()


def cmd_bootstrap(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    result = run_bootstrap(
        settings=settings,
        base_dir=base_dir,
        checkpoint=args.checkpoint,
        teacher=args.teacher,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        teacher_device=args.teacher_device,
        teacher_quant=args.teacher_quant,
        teacher_max_memory=args.teacher_max_memory,
        reuse_dataset=args.reuse_dataset,
        batch_tokens=args.batch_tokens,
        grad_accum_steps=args.grad_accum,
        profile_name=args.profile,
    )
    print(result)


def cmd_ingest_once(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    allowlist = _resolve_allowlist(settings)
    new_docs = ingest_sources(base_dir, allowlist, settings)
    print({"ok": True, "new_docs": new_docs, "knowledge_path": settings.get("continuous", {}).get("knowledge_path")})


def cmd_train_once(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    max_steps_env = os.getenv("C3RNT2_TRAIN_MAX_STEPS")
    max_steps_val = None
    if max_steps_env:
        try:
            parsed = int(max_steps_env)
        except Exception:
            parsed = None
        if parsed is not None and parsed > 0:
            max_steps_val = parsed
    base_dir = Path(".")
    try:
        lock = acquire_exclusive_lock(base_dir, "train")
    except LockUnavailable:
        print({"ok": False, "error": "train lock unavailable (serve/self_patch running?)"})
        return
    try:
        result = train_once_backend(settings, base_dir, reuse_dataset=bool(args.reuse_dataset), max_steps=max_steps_val)
        payload = dict(result) if isinstance(result, dict) else dict(getattr(result, "__dict__", {}) or {})
        adapter_dir = payload.get("adapter_dir")
        if adapter_dir is not None:
            payload["adapter_dir"] = str(adapter_dir)
        print(json.dumps(payload, ensure_ascii=True))
        if not bool(payload.get("ok", False)):
            sys.exit(1)
    finally:
        lock.release()


def cmd_self_train(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    interval_min = _resolve_interval_minutes(args.interval_minutes, settings)
    once = bool(args.once)
    while True:
        allowlist = _resolve_allowlist(settings)
        new_docs = ingest_sources(base_dir, allowlist, settings)
        try:
            lock = acquire_exclusive_lock(base_dir, "train")
        except LockUnavailable:
            print({"ok": False, "error": "train lock unavailable (serve/self_patch running?)"})
            if once:
                return
            time.sleep(max(5.0, interval_min * 60.0))
            continue
        try:
            strategy = str((settings.get("server", {}) or {}).get("train_strategy", "subprocess")).lower()
            if strategy.startswith("subprocess"):
                train_result = _run_train_subprocess(settings, reuse_dataset=bool(args.reuse_dataset))
            else:
                train_result = train_once_backend(settings, base_dir, reuse_dataset=bool(args.reuse_dataset))
            payload = dict(train_result) if isinstance(train_result, dict) else dict(getattr(train_result, "__dict__", {}) or {})
            adapter_dir = payload.get("adapter_dir")
            if adapter_dir is not None:
                payload["adapter_dir"] = str(adapter_dir)
            print({"ingest_new_docs": new_docs, "train": payload})
        finally:
            lock.release()
        if once:
            break
        time.sleep(max(5.0, interval_min * 60.0))


def _run_self_train_tick(
    app: object,
    settings: dict,
    base_dir: Path,
    *,
    reuse_dataset: bool,
    maintenance_window_s: float,
    reload_fn: Callable | None = None,
    train_fn: Callable | None = None,
) -> dict:
    allowlist = _resolve_allowlist(settings)
    cont_cfg = settings.get("continuous", {}) or {}
    if bool(cont_cfg.get("ingest_web", True)) and not allowlist:
        return {"ok": False, "error": "ingest_web enabled but allow_domains empty", "skipped": "web_ingest_unavailable"}
    web_ok, web_msg, strict = _check_web_ingest(settings, allowlist)
    if not web_ok:
        if strict:
            return {"ok": False, "error": web_msg or "web_ingest_invalid", "skipped": "web_ingest_unavailable"}
        print(f"WARN {web_msg}")
    try:
        new_docs = ingest_sources(base_dir, allowlist, settings)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc), "skipped": "web_ingest_unavailable"}

    # If web ingest is misconfigured but not strict, continue (best-effort) using local sources.
    # This keeps self-train/serve-self-train usable in offline/default setups.

    collected = collect_samples(base_dir, allowlist, settings, ingest=False)
    stats = collected.stats
    stats.new_docs = int(new_docs)
    trigger_cfg = settings.get("continuous", {}).get("trigger", {})
    if bool(trigger_cfg.get("enabled", True)):
        min_new_docs = int(trigger_cfg.get("min_new_docs", 1))
        min_novelty = float(trigger_cfg.get("min_novelty", 0.2))
        min_successes = int(trigger_cfg.get("min_successes", 0))
        if stats.new_docs < min_new_docs and stats.novelty_avg < min_novelty and stats.successes <= min_successes:
            return {"ok": True, "skipped": "signal_low", "new_docs": new_docs, "stats": stats.__dict__}

    budgets_cfg = (settings.get("continuous", {}) or {}).get("budgets", {}) or {}
    try:
        from .continuous.budgets import can_start_run

        planned_steps = (settings.get("continuous", {}) or {}).get("max_steps_per_tick", (settings.get("continuous", {}) or {}).get("max_steps"))
        decision = can_start_run(
            base_dir,
            max_steps_per_day=budgets_cfg.get("max_steps_per_day"),
            max_tokens_per_day=budgets_cfg.get("max_tokens_per_day"),
            max_disk_mb=budgets_cfg.get("max_disk_mb"),
            planned_steps=int(planned_steps) if planned_steps is not None else None,
        )
        if not decision.ok:
            payload = {"ok": False, "error": f"budget_limit:{decision.reason}", "budget": decision.state, "budget_state_path": str(decision.state_path)}
            print({"self_train_budget": payload})
            return {**payload, "skipped": "budget_limit"}
    except Exception:
        # Never block self-train on budget module errors.
        pass

    server_cfg = settings.get("server", {}) or {}
    block_during_training = bool(server_cfg.get("block_during_training", False))
    train_strategy = str(server_cfg.get("train_strategy", "subprocess")).lower()
    lock_path = base_dir / "data" / "locks" / "train.lock"
    lock = FileLock(lock_path)
    did_unload = False
    try:
        lock.acquire(blocking=False)
    except LockUnavailable:
        return {"ok": False, "error": "train_lock_unavailable", "new_docs": new_docs}
    try:
        state = getattr(app, "state", SimpleNamespace())
        setattr(app, "state", state)
        state.training_active = True
        if maintenance_window_s and maintenance_window_s > 0 and not block_during_training:
            state.maintenance_until = time.time() + float(maintenance_window_s)
        try:
            if train_strategy in {"unload_reload", "subprocess_unload"}:
                _unload_models_for_train(app)
                did_unload = True
            if train_strategy in {"subprocess", "subprocess_unload"}:
                timeout_s = None
                max_walltime_min = budgets_cfg.get("max_walltime_min")
                try:
                    if max_walltime_min is not None and float(max_walltime_min) > 0:
                        timeout_s = float(max_walltime_min) * 60.0
                except Exception:
                    timeout_s = None
                result_dict = _run_train_subprocess(settings, reuse_dataset=reuse_dataset, timeout_s=timeout_s)
                result = _coerce_train_result(result_dict)
            else:
                result = _run_train_inprocess(settings, base_dir, reuse_dataset, train_fn=train_fn)
        except Exception as exc:
            result = None
            error = str(exc)
        else:
            error = None
    finally:
        lock.release()
    reload_error = None
    if did_unload:
        try:
            _reload_models_for_app(app, settings)
        except Exception as exc:
            reload_error = f"reload_model_failed: {exc}"
    try:
        if reload_error:
            # Keep the server blocked if we failed to restore the model.
            app.state.training_active = True
        else:
            app.state.training_active = False
        if maintenance_window_s and maintenance_window_s > 0:
            app.state.maintenance_until = time.time() + float(maintenance_window_s)
    except Exception:
        pass
    if error:
        message = error
        if reload_error:
            message = f"{message}; {reload_error}"
        return {"ok": False, "error": message, "new_docs": new_docs, "stats": stats.__dict__}
    if reload_error:
        return {"ok": False, "error": reload_error, "new_docs": new_docs, "stats": stats.__dict__}
    reload_result = None
    if reload_fn is not None and result and bool(getattr(result, "ok_eval", getattr(result, "eval_ok", getattr(result, "ok", False)))):
        try:
            reload_result = reload_fn(app, base_dir, settings, force=True)
        except Exception as exc:
            reload_result = {"ok": False, "error": str(exc)}
    metrics = {
        "docs_new": new_docs,
        "samples_used": getattr(result, "samples", None) if result else None,
        "steps": getattr(result, "steps", None) if result else None,
        "loss": getattr(result, "loss", None) if result else None,
        "tokens_per_sec": getattr(result, "tokens_per_sec", None) if result else None,
        "vram_peak_mb": getattr(result, "vram_peak_mb", None) if result else None,
        "eval_ok": getattr(result, "eval_ok", None) if result else None,
    }
    try:
        from .continuous.budgets import record_run

        steps_val = int(getattr(result, "steps", 0) or 0) if result else 0
        tokens_val = int(getattr(result, "tokens_seen", getattr(result, "tokens", 0)) or 0) if result else 0
        if steps_val > 0 or tokens_val > 0:
            record = record_run(base_dir, steps=steps_val, tokens=tokens_val)
            metrics["budget_recorded"] = bool(record.ok)
            metrics["budget_state"] = record.state
    except Exception:
        pass
    print({"self_train_metrics": metrics})
    return {"ok": True, "new_docs": new_docs, "train": result.__dict__ if result else None, "reload": reload_result, "stats": stats.__dict__}


def cmd_serve_self_train(args: argparse.Namespace) -> None:
    settings = _load_and_validate(
        args.profile,
        override=lambda s: _apply_serve_self_train_defaults(_apply_cli_overrides(s, args)),
    )
    base_dir = Path(".")
    interval_min = _resolve_interval_minutes(args.interval_minutes, settings)
    maintenance_window_s = float(args.maintenance_window_s or settings.get("server", {}).get("maintenance_window_s", 10))
    reuse_dataset = bool(args.reuse_dataset)
    once = bool(args.once)
    reload_fn = None
    app = None
    train_fn = None

    if args.mock:
        settings.setdefault("server", {})["train_strategy"] = "inprocess"
        cont = settings.setdefault("continuous", {})
        cont["ingest_web"] = False
        cont.setdefault("trigger", {})["enabled"] = False
        class _DummyLock:
            def read_lock(self):
                return nullcontext()

            def write_lock(self):
                return nullcontext()

        app = SimpleNamespace(state=SimpleNamespace(model_lock=_DummyLock(), models={}, model=None))
        dummy_result = SimpleNamespace(
            ok=True,
            ok_train=True,
            ok_eval=True,
            eval_ok=True,
            adapter_dir=None,
            loss=0.0,
            steps=1,
            samples=1,
            tokens_per_sec=0.0,
            vram_peak_mb=None,
        )

        def _mock_train(_settings, _base_dir, reuse_dataset: bool = False):
            _ = _settings, _base_dir, reuse_dataset
            return dummy_result

        def _mock_reload(_app, _base_dir, _settings, force: bool = False):
            return {"ok": True, "reloaded": bool(force), "mock": True}
        reload_fn = _mock_reload
        train_fn = _mock_train
    else:
        try:
            from .server import create_app, reload_latest_adapter_for_app
        except Exception as exc:
            print({"ok": False, "error": f"server_unavailable: {exc}"})
            return
        app = create_app(settings, base_dir=base_dir)
        reload_fn = reload_latest_adapter_for_app

        def _serve():
            import uvicorn  # type: ignore

            uvicorn.run(app, host=args.host, port=args.port, log_level="info")

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()

    while True:
        result = _run_self_train_tick(
            app,
            settings,
            base_dir,
            reuse_dataset=reuse_dataset,
            maintenance_window_s=maintenance_window_s,
            reload_fn=reload_fn,
            train_fn=train_fn,
        )
        print({"self_train": result})
        if once:
            break
        time.sleep(max(5.0, interval_min * 60.0))


def _run_autopilot_train_with_server(
    app: object,
    settings: dict,
    base_dir: Path,
    *,
    profile: str,
    reuse_dataset: bool,
    max_steps: int | None,
) -> dict:
    server_cfg = settings.get("server", {}) or {}
    train_strategy = str(server_cfg.get("train_strategy", "subprocess")).lower()
    block_during_training = bool(server_cfg.get("block_during_training", False))
    maintenance_window_s = float(server_cfg.get("maintenance_window_s", 10))

    state = getattr(app, "state", SimpleNamespace())
    setattr(app, "state", state)

    did_unload = False
    reload_error = None
    payload: dict = {"ok": False, "ok_train": False, "ok_eval": False, "error": "train_not_started"}
    try:
        state.training_active = True
        if maintenance_window_s and maintenance_window_s > 0 and not block_during_training:
            state.maintenance_until = time.time() + float(maintenance_window_s)

        if train_strategy in {"unload_reload", "subprocess_unload"}:
            _unload_models_for_train(app)
            did_unload = True

        if train_strategy in {"subprocess", "subprocess_unload"}:
            cmd = [sys.executable, "-m", "c3rnt2", "train-once", "--profile", str(profile)]
            if reuse_dataset:
                cmd.append("--reuse-dataset")
            env = dict(os.environ)
            if max_steps is not None and int(max_steps) > 0:
                env["C3RNT2_TRAIN_MAX_STEPS"] = str(int(max_steps))
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                payload = {"ok": False, "ok_train": False, "ok_eval": False, "error": result.stderr.strip() or "train subprocess failed"}
            else:
                parsed = None
                for line in reversed(result.stdout.splitlines()):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            parsed = json.loads(line)
                            break
                        except Exception:
                            continue
                payload = parsed if isinstance(parsed, dict) else {"ok": False, "ok_train": False, "ok_eval": False, "error": "train subprocess output not parseable"}
        else:
            lock_path = base_dir / "data" / "locks" / "train.lock"
            tlock = FileLock(lock_path)
            try:
                tlock.acquire(blocking=False)
            except LockUnavailable:
                payload = {"ok": False, "ok_train": False, "ok_eval": False, "error": "train_lock_unavailable"}
            else:
                try:
                    result_obj = train_once_backend(settings, base_dir, reuse_dataset=reuse_dataset, max_steps=max_steps)
                    payload = dict(result_obj) if isinstance(result_obj, dict) else dict(getattr(result_obj, "__dict__", {}) or {})
                    adapter_dir = payload.get("adapter_dir")
                    if adapter_dir is not None:
                        payload["adapter_dir"] = str(adapter_dir)
                except Exception as exc:
                    payload = {"ok": False, "ok_train": False, "ok_eval": False, "error": str(exc)}
                finally:
                    tlock.release()
    finally:
        if did_unload:
            try:
                _reload_models_for_app(app, settings)
            except Exception as exc:
                reload_error = f"reload_model_failed: {exc}"
        try:
            if reload_error:
                state.training_active = True
            else:
                state.training_active = False
            if maintenance_window_s and maintenance_window_s > 0:
                state.maintenance_until = time.time() + float(maintenance_window_s)
        except Exception:
            pass

    if reload_error:
        prev = str(payload.get("error") or "")
        payload["ok"] = False
        payload["error"] = f"{prev}; {reload_error}".strip("; ").strip()
    return payload


def cmd_serve_autopilot(args: argparse.Namespace) -> None:
    settings = _load_and_validate(
        args.profile,
        override=lambda s: _apply_serve_self_train_defaults(_apply_cli_overrides(s, args)),
    )
    base_dir = Path(".")
    interval_min = float(args.interval_minutes) if args.interval_minutes is not None else float((settings.get("autopilot", {}) or {}).get("interval_minutes", settings.get("continuous", {}).get("interval_minutes", 30)))
    once = bool(args.once)
    no_web = bool(args.no_web) or bool(args.mock)
    force = bool(args.force)

    if args.mock:
        settings.setdefault("server", {})["train_strategy"] = "inprocess"

        class _DummyLock:
            def read_lock(self):
                return nullcontext()

            def write_lock(self):
                return nullcontext()

        app = SimpleNamespace(state=SimpleNamespace(model_lock=_DummyLock(), models={}, model=None, training_active=False, maintenance_until=0.0))
    else:
        try:
            from .server import create_app
        except Exception as exc:
            print({"ok": False, "error": f"server_unavailable: {exc}"})
            return
        app = create_app(settings, base_dir=base_dir)

        def _serve():
            import uvicorn  # type: ignore

            uvicorn.run(app, host=args.host, port=args.port, log_level="info")

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()

    profile_name = settings.get("_profile") or os.getenv("C3RNT2_PROFILE") or resolve_profile(None)

    def _train_runner(_profile: str, reuse_dataset: bool, max_steps: int | None):
        # Ignore the incoming profile arg and use the resolved one for consistency.
        _ = _profile
        return _run_autopilot_train_with_server(
            app,
            settings,
            base_dir,
            profile=str(profile_name),
            reuse_dataset=reuse_dataset,
            max_steps=max_steps,
        )

    while True:
        result = run_autopilot_tick(
            settings,
            base_dir,
            no_web=no_web,
            mock=bool(args.mock),
            force=force,
            train_runner=_train_runner,
        )
        print({"serve_autopilot": {"ok": result.ok, "steps": result.steps, "error": result.error}})
        if once:
            break
        time.sleep(max(5.0, interval_min * 60.0))


def cmd_self_patch(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    try:
        lock = acquire_exclusive_lock(base_dir, "self_patch")
    except LockUnavailable:
        print({"ok": False, "error": "self_patch lock unavailable (serve/train running?)"})
        return
    try:
        from .self_patch.propose_patch import propose_patch
        from .self_patch.sandbox_run import sandbox_run
        from .self_patch.apply_patch import apply_patch

        diff_text = None
        if args.diff_file:
            diff_text = Path(args.diff_file).read_text(encoding="utf-8")
        proposal = propose_patch(
            args.goal,
            {"changes": {}},
            base_dir,
            settings=settings,
            dry_run=args.dry_run,
            diff_text=diff_text,
        )
        sandbox = sandbox_run(base_dir, proposal.patch_id, bench=not args.no_bench, settings=settings, allow_empty=True)
        apply_result = None
        if args.approve:
            try:
                meta = json.loads(proposal.meta_path.read_text(encoding="utf-8"))
                meta["ready_for_review"] = True
                meta["status"] = "ready_for_review"
                proposal.meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
            except Exception:
                pass
            apply_result = apply_patch(proposal.patch_id, base_dir, settings=settings)
        print(
            {
                "patch_id": proposal.patch_id,
                "sandbox_ok": bool(sandbox.get("ok")),
                "apply_ok": getattr(apply_result, "ok", None) if apply_result else None,
                "apply_error": getattr(apply_result, "error", None) if apply_result else None,
                "queue_dir": str(proposal.queue_dir),
            }
        )
    finally:
        lock.release()


def cmd_apply_patch(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    if not args.approve:
        print({"ok": False, "error": "approve flag required"})
        sys.exit(1)
    try:
        lock = acquire_exclusive_lock(base_dir, "self_patch")
    except LockUnavailable:
        print({"ok": False, "error": "self_patch lock unavailable (serve/train running?)"})
        return
    try:
        from .self_patch.apply_patch import apply_patch

        queue_dir = _self_patch_queue_dir(base_dir, settings) / args.patch_id
        meta_path = queue_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["ready_for_review"] = True
                meta["status"] = "ready_for_review"
                meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")
            except Exception:
                pass
        result = apply_patch(args.patch_id, base_dir, settings=settings)
        print({"ok": result.ok, "error": result.error, "patch_id": result.patch_id})
        if not result.ok:
            sys.exit(1)
    finally:
        lock.release()


def cmd_eval(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    backend = str(settings.get("core", {}).get("backend", "vortex")).lower()
    if backend == "hf":
        if torch is None:
            print({"ok": False, "error": "torch not available"})
            sys.exit(1)
        model = load_inference_model(settings)
        if not hasattr(model, "model") or not hasattr(model, "tokenizer"):
            print({"ok": False, "error": "hf model not available"})
            sys.exit(1)
        samples = ["hello world", "def add(a, b):", "json: {\"k\": 1}"]
        losses = []
        with torch.inference_mode():
            for text in samples:
                enc = model.tokenizer(text, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model.model(**enc, labels=enc["input_ids"])
                losses.append(float(out.loss.item()))
        ppl = float(torch.exp(torch.tensor(losses).mean()).item()) if losses else None
        print({"backend": "hf", "loss": sum(losses) / max(1, len(losses)), "ppl": ppl})
        return
    # Core eval: generate and report tokens/sec
    core = load_inference_model(settings)
    start = time.time()
    _ = core.generate("def f(x):", max_new_tokens=32)
    tps = 32 / max(1e-6, time.time() - start)
    print({"backend": "vortex", "tokens_per_sec": round(tps, 3)})


def cmd_agent_demo(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    report = run_demo_agent(settings)
    print(report)


def cmd_agent_run(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    report = run_agent(args.task, settings, base_dir, max_iters=int(args.max_iters))
    print(report)


def cmd_learn_ingest(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    result = collect_from_episodes(base_dir, settings, max_events=args.max_events)
    print(result.__dict__)
    if not result.ok:
        sys.exit(1)


def cmd_learn_train(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    result = train_qlora(settings, base_dir, steps=args.steps, reuse_dataset=False)
    print(result.__dict__ if hasattr(result, "__dict__") else result)
    if hasattr(result, "ok") and not result.ok:
        sys.exit(1)


def cmd_learn_eval(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    adapter = Path(args.adapter) if args.adapter else None
    if adapter and not adapter.is_absolute():
        adapter = base_dir / adapter
    result = evaluate_adapter(base_dir, settings, adapter_path=adapter)
    log_eval(base_dir, settings, result)
    print(result.__dict__)
    if not result.ok:
        sys.exit(1)


def cmd_learn_promote(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    result = promote_latest(base_dir, settings, min_improvement=args.min_improvement)
    print(result.__dict__)
    if not result.ok:
        sys.exit(1)


def cmd_promote_quarantine(args: argparse.Namespace) -> None:
    base_dir = Path(".")
    result = promote_quarantine_run(base_dir, run_id=str(args.run_id), require_approval=not bool(args.no_approval))
    payload = dict(getattr(result, "__dict__", {}) or {})
    print(json.dumps(payload, ensure_ascii=True))
    if not bool(payload.get("ok", False)):
        sys.exit(1)


def cmd_tokenizer_train(args: argparse.Namespace) -> None:
    _run_module("c3rnt2.tokenizer.rnt2_train", args.extra)


def cmd_train_experts(args: argparse.Namespace) -> None:
    _run_module("c3rnt2.training.train_experts", args.extra)


def cmd_train_router(args: argparse.Namespace) -> None:
    _run_module("c3rnt2.training.train_router", args.extra)


def cmd_finetune_adapter(args: argparse.Namespace) -> None:
    _run_module("c3rnt2.training.finetune_adapters", args.extra)


def cmd_self_improve(args: argparse.Namespace) -> None:
    # Backward-compatible alias for self-patch.
    args.goal = args.goal or "self-improve"
    cmd_self_patch(args)


def cmd_autopilot(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    run_autopilot_loop(
        settings,
        base_dir,
        once=bool(args.once),
        interval_minutes=args.interval_minutes,
        no_web=bool(args.no_web),
        mock=bool(args.mock),
        force=bool(args.force),
    )


def cmd_autopatch_once(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    patch_device = os.getenv("C3RNT2_PATCH_DEVICE", "").strip().lower()
    if patch_device:
        core = dict(settings.get("core", {}) or {})
        if patch_device in {"cpu"}:
            core["hf_device"] = "cpu"
            core["device"] = "cpu"
        settings["core"] = core
    try:
        lock = acquire_exclusive_lock(base_dir, "self_patch")
    except LockUnavailable:
        print(json.dumps({"ok": False, "promoted": False, "error": "self_patch lock unavailable"}, ensure_ascii=True))
        sys.exit(1)
    try:
        eval_short = {"ok": False} if bool(getattr(args, "eval_regression", False)) else None
        result = run_autopatch_once(settings, base_dir, profile=resolve_profile(args.profile), eval_short=eval_short)
        print(json.dumps(result, ensure_ascii=True))
        if not result.get("ok", False):
            sys.exit(1)
    finally:
        lock.release()

def _update_bench_baseline(bench_dir: Path, bench: dict) -> dict:
    profile = str(bench.get("profile") or "")
    backend = str(bench.get("backend") or "")
    if not profile or not backend:
        return {"ok": False, "error": "missing_profile_or_backend"}
    bench_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = bench_dir / "baseline.json"
    baseline: dict[str, dict[str, dict]] = {}
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8")) or {}
        except Exception:
            baseline = {}
    prof_entry = dict(baseline.get(profile, {}) or {})
    created = False
    if backend not in prof_entry:
        prof_entry[backend] = dict(bench)
        baseline[profile] = prof_entry
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=True, indent=2), encoding="utf-8")
        created = True
    return {"ok": True, "baseline_created": created}


def cmd_bench(args: argparse.Namespace) -> None:
    from .bench import BenchArgs, run_bench

    profile = args.profile or resolve_profile(None)
    settings = _load_and_validate(profile)
    base_dir = Path(".")
    prompt_text = str(getattr(args, "prompt", "") or "")
    prompt_file = getattr(args, "prompt_file", None)
    prompt_file_str = str(prompt_file) if prompt_file else None
    if prompt_file:
        try:
            prompt_text = Path(str(prompt_file)).read_text(encoding="utf-8")
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"prompt_file_read_failed:{exc}"}, ensure_ascii=True))
            sys.exit(1)

    json_out = Path(str(getattr(args, "json_out", "data/bench/last.json")))
    jsonl_out_raw = getattr(args, "jsonl_out", None)
    jsonl_out = Path(str(jsonl_out_raw)) if jsonl_out_raw else None
    max_new = int(getattr(args, "max_new", getattr(args, "max_new_tokens", 64)) or 64)
    ctx = getattr(args, "ctx", None)
    ctx_val = int(ctx) if ctx is not None else None

    report = run_bench(
        settings,
        base_dir=base_dir,
        args=BenchArgs(
            profile=str(profile),
            prompt=prompt_text,
            prompt_file=prompt_file_str,
            ctx=ctx_val,
            max_new=max_new,
            warmup=int(getattr(args, "warmup", 1) or 0),
            repeat=int(getattr(args, "repeat", 3) or 1),
            seed=int(getattr(args, "seed", 0) or 0),
            json_out=json_out,
            jsonl_out=jsonl_out,
        ),
    )
    print(json.dumps(report, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="c3rnt2")
    sub = parser.add_subparsers(dest="command")

    doc = sub.add_parser("doctor")
    doc.add_argument("--profile", default=None)
    doc.add_argument("--deep", action="store_true")
    doc.add_argument("--mock", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    chat = sub.add_parser("chat")
    chat.add_argument("--profile", default=None)
    chat.add_argument("--backend", default=None)
    chat.add_argument("--model", default=None)
    chat.add_argument("--device", default=None)
    chat.add_argument("--stream", action="store_true")
    chat.add_argument("--max-new-tokens", type=int, default=None)
    chat.add_argument("--temperature", type=float, default=None)
    chat.add_argument("--top-p", type=float, default=None)
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve")
    serve.add_argument("--profile", default=None)
    serve.add_argument("--backend", default=None)
    serve.add_argument("--model", default=None)
    serve.add_argument("--device", default=None)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    serve_self = sub.add_parser("serve-self-train")
    serve_self.add_argument("--profile", default=None)
    serve_self.add_argument("--backend", default=None)
    serve_self.add_argument("--model", default=None)
    serve_self.add_argument("--device", default=None)
    serve_self.add_argument("--host", default="0.0.0.0")
    serve_self.add_argument("--port", type=int, default=8000)
    serve_self.add_argument("--once", action="store_true")
    serve_self.add_argument("--interval-minutes", type=float, default=None)
    serve_self.add_argument("--reuse-dataset", action="store_true")
    serve_self.add_argument("--maintenance-window-s", type=float, default=None)
    serve_self.add_argument("--mock", action="store_true")
    serve_self.set_defaults(func=cmd_serve_self_train)

    serve_auto = sub.add_parser("serve-autopilot")
    serve_auto.add_argument("--profile", default=None)
    serve_auto.add_argument("--backend", default=None)
    serve_auto.add_argument("--model", default=None)
    serve_auto.add_argument("--device", default=None)
    serve_auto.add_argument("--host", default="0.0.0.0")
    serve_auto.add_argument("--port", type=int, default=8000)
    serve_auto.add_argument("--once", action="store_true")
    serve_auto.add_argument("--interval-minutes", type=float, default=None)
    serve_auto.add_argument("--no-web", action="store_true")
    serve_auto.add_argument("--mock", action="store_true")
    serve_auto.add_argument("--force", action="store_true")
    serve_auto.set_defaults(func=cmd_serve_autopilot)

    bench = sub.add_parser("bench")
    bench.add_argument("--profile", default=None)
    bench.add_argument("--prompt-file", default=None)
    bench.add_argument("--prompt", default="def f(x):\n    return x\n")
    bench.add_argument("--ctx", type=int, default=None)
    bench.add_argument("--max-new", dest="max_new", type=int, default=64)
    bench.add_argument("--max-new-tokens", dest="max_new", type=int, default=None)
    bench.add_argument("--repeat", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--json-out", default="data/bench/last.json")
    bench.add_argument("--jsonl-out", default=None)
    bench.set_defaults(func=cmd_bench)

    boot = sub.add_parser("bootstrap")
    boot.add_argument("--profile", default=None)
    boot.add_argument("--checkpoint", default=None)
    boot.add_argument("--teacher", default=None)
    boot.add_argument("--teacher-device", default="cuda")
    boot.add_argument("--teacher-quant", default="none", choices=["none", "8bit", "4bit"])
    boot.add_argument("--teacher-max-memory", default=None)
    boot.add_argument("--max-prompts", type=int, default=16)
    boot.add_argument("--max-new-tokens", type=int, default=64)
    boot.add_argument("--steps", type=int, default=50)
    boot.add_argument("--reuse-dataset", action="store_true")
    boot.add_argument("--batch-tokens", type=int, default=4096)
    boot.add_argument("--grad-accum", type=int, default=1)
    boot.set_defaults(func=cmd_bootstrap)

    ingest = sub.add_parser("ingest-once")
    ingest.add_argument("--profile", default=None)
    ingest.set_defaults(func=cmd_ingest_once)

    train = sub.add_parser("train-once")
    train.add_argument("--profile", default=None)
    train.add_argument("--reuse-dataset", action="store_true")
    train.set_defaults(func=cmd_train_once)

    self_train = sub.add_parser("self-train")
    self_train.add_argument("--profile", default=None)
    self_train.add_argument("--once", action="store_true")
    self_train.add_argument("--interval-minutes", type=float, default=None)
    self_train.add_argument("--reuse-dataset", action="store_true")
    self_train.set_defaults(func=cmd_self_train)

    sp = sub.add_parser("self-patch")
    sp.add_argument("--profile", default=None)
    sp.add_argument("--goal", required=True)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--approve", action="store_true")
    sp.add_argument("--diff-file", default=None)
    sp.add_argument("--no-bench", action="store_true")
    sp.set_defaults(func=cmd_self_patch)

    ap = sub.add_parser("apply-patch")
    ap.add_argument("patch_id")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--approve", action="store_true")
    ap.set_defaults(func=cmd_apply_patch)

    ev = sub.add_parser("eval")
    ev.add_argument("--profile", default=None)
    ev.set_defaults(func=cmd_eval)

    ad = sub.add_parser("agent-demo")
    ad.add_argument("--profile", default=None)
    ad.set_defaults(func=cmd_agent_demo)

    ar = sub.add_parser("agent-run")
    ar.add_argument("--profile", default=None)
    ar.add_argument("--task", required=True)
    ar.add_argument("--max-iters", type=int, default=5)
    ar.set_defaults(func=cmd_agent_run)

    tok = sub.add_parser("tokenizer-train")
    tok.add_argument("extra", nargs=argparse.REMAINDER)
    tok.set_defaults(func=cmd_tokenizer_train)

    te = sub.add_parser("train-experts")
    te.add_argument("extra", nargs=argparse.REMAINDER)
    te.set_defaults(func=cmd_train_experts)

    tr = sub.add_parser("train-router")
    tr.add_argument("extra", nargs=argparse.REMAINDER)
    tr.set_defaults(func=cmd_train_router)

    fa = sub.add_parser("finetune-adapter")
    fa.add_argument("extra", nargs=argparse.REMAINDER)
    fa.set_defaults(func=cmd_finetune_adapter)

    si = sub.add_parser("self-improve")
    si.add_argument("--profile", default=None)
    si.add_argument("--goal", default=None)
    si.add_argument("--dry-run", action="store_true")
    si.add_argument("--approve", action="store_true")
    si.add_argument("--diff-file", default=None)
    si.add_argument("--no-bench", action="store_true")
    si.set_defaults(func=cmd_self_improve)

    ap = sub.add_parser("autopilot")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-minutes", type=float, default=None)
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.set_defaults(func=cmd_autopilot)

    apo = sub.add_parser("autopatch-once")
    apo.add_argument("--profile", default=None)
    apo.add_argument("--eval-regression", action="store_true")
    apo.set_defaults(func=cmd_autopatch_once)

    learn = sub.add_parser("learn")
    learn_sub = learn.add_subparsers(dest="learn_command")

    learn_ingest = learn_sub.add_parser("ingest")
    learn_ingest.add_argument("--profile", default=None)
    learn_ingest.add_argument("--max-events", type=int, default=None)
    learn_ingest.set_defaults(func=cmd_learn_ingest)

    learn_train = learn_sub.add_parser("train")
    learn_train.add_argument("--profile", default=None)
    learn_train.add_argument("--steps", type=int, default=None)
    learn_train.set_defaults(func=cmd_learn_train)

    learn_eval = learn_sub.add_parser("eval")
    learn_eval.add_argument("--profile", default=None)
    learn_eval.add_argument("--adapter", default=None)
    learn_eval.set_defaults(func=cmd_learn_eval)

    learn_promote = learn_sub.add_parser("promote")
    learn_promote.add_argument("--profile", default=None)
    learn_promote.add_argument("--min-improvement", type=float, default=None)
    learn_promote.set_defaults(func=cmd_learn_promote)

    promo = sub.add_parser("promote-quarantine")
    promo.add_argument("--run-id", required=True)
    promo.add_argument("--no-approval", action="store_true")
    promo.set_defaults(func=cmd_promote_quarantine)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if args.command == "learn" and not getattr(args, "learn_command", None):
        learn.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
