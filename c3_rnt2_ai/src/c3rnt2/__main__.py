from __future__ import annotations

import argparse
import json
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

from .config import load_settings, resolve_profile, validate_profile
from .continuous.dataset import ingest_sources
from .continuous.bootstrap import run_bootstrap
from .device import detect_device
from .doctor import check_deps, run_deep_checks
from .model_loader import load_inference_model
from .prompting.chat_format import build_chat_prompt
from .server import run_server
from .utils.locks import LockUnavailable, acquire_exclusive_lock, FileLock
from .agent.agent_loop import run_demo_agent
from .training.hf_qlora import train_once as train_hf_once
from .utils.oom import is_oom_error, clear_cuda_cache
from .learning_loop.data_collector import collect_from_episodes
from .learning_loop.data_curator import curate_dataset
from .learning_loop.trainer import train_qlora
from .learning_loop.evaluator import evaluate_adapter, log_eval
from .learning_loop.promoter import promote_latest
from .agent.runner import run_agent


def _load_and_validate(profile: str | None, override: Callable[[dict], dict] | None = None) -> dict:
    settings = load_settings(profile)
    if override is not None:
        settings = override(settings)
    validate_profile(settings, base_dir=Path("."))
    return settings


def _resolve_allowlist(settings: dict) -> list[str]:
    tools_cfg = settings.get("tools", {}) or {}
    web_cfg = tools_cfg.get("web", {}) or {}
    if web_cfg.get("allow_domains"):
        return list(web_cfg.get("allow_domains"))
    agent_cfg = settings.get("agent", {}) or {}
    return list(agent_cfg.get("web_allowlist", []))


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


def _run_module(module: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module] + extra_args
    subprocess.run(cmd, check=True)


def cmd_doctor(args: argparse.Namespace) -> None:
    info = detect_device()
    print(
        {
            "device": info.device,
            "cuda_available": info.cuda_available,
            "gpu": info.name,
            "vram_gb": info.vram_gb,
            "dtype": info.dtype,
            "python": sys.version.split()[0],
        }
    )
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
        if not args.deep:
            return
        settings = load_settings(args.profile)

    if args.deep:
        try:
            deep_result = run_deep_checks(settings, base_dir=base_dir)
            print({"deep": deep_result})
        except Exception as exc:
            print({"deep": {"deep_ok": False, "error": str(exc)}})


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
    base_dir = Path(".")
    try:
        lock = acquire_exclusive_lock(base_dir, "train")
    except LockUnavailable:
        print({"ok": False, "error": "train lock unavailable (serve/self_patch running?)"})
        return
    try:
        result = train_hf_once(settings, base_dir, reuse_dataset=args.reuse_dataset)
        print(result.__dict__)
        if not result.ok:
            sys.exit(1)
    finally:
        lock.release()


def cmd_self_train(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    interval_min = float(args.interval_minutes or settings.get("continuous", {}).get("run_interval_minutes", 30))
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
            train_result = train_hf_once(settings, base_dir, reuse_dataset=args.reuse_dataset)
            print({"ingest_new_docs": new_docs, "train": train_result.__dict__})
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
) -> dict:
    allowlist = _resolve_allowlist(settings)
    new_docs = ingest_sources(base_dir, allowlist, settings)
    lock_path = base_dir / "data" / "locks" / "train.lock"
    lock = FileLock(lock_path)
    try:
        lock.acquire(blocking=False)
    except LockUnavailable:
        return {"ok": False, "error": "train_lock_unavailable", "new_docs": new_docs}
    try:
        state = getattr(app, "state", SimpleNamespace())
        setattr(app, "state", state)
        state.training_active = True
        if maintenance_window_s and maintenance_window_s > 0:
            state.maintenance_until = time.time() + float(maintenance_window_s)
        result = train_hf_once(settings, base_dir, reuse_dataset=reuse_dataset)
    finally:
        try:
            app.state.training_active = False
        except Exception:
            pass
        lock.release()
    reload_result = None
    if reload_fn is not None and result.ok:
        try:
            reload_result = reload_fn(app, base_dir, settings, force=True)
        except Exception as exc:
            reload_result = {"ok": False, "error": str(exc)}
    return {"ok": True, "new_docs": new_docs, "train": result.__dict__, "reload": reload_result}


def cmd_serve_self_train(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile, override=lambda s: _apply_cli_overrides(s, args))
    base_dir = Path(".")
    interval_min = float(args.interval_minutes or settings.get("continuous", {}).get("run_interval_minutes", 30))
    maintenance_window_s = float(args.maintenance_window_s or settings.get("server", {}).get("maintenance_window_s", 10))
    reuse_dataset = bool(args.reuse_dataset)
    once = bool(args.once)
    reload_fn = None
    app = None

    if args.mock:
        class _DummyLock:
            def read_lock(self):
                return nullcontext()

            def write_lock(self):
                return nullcontext()

        app = SimpleNamespace(state=SimpleNamespace(model_lock=_DummyLock(), models={}, model=None))
        def _mock_reload(_app, _base_dir, _settings, force: bool = False):
            return {"ok": True, "reloaded": bool(force), "mock": True}
        reload_fn = _mock_reload
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
        )
        print({"self_train": result})
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

def cmd_bench(args: argparse.Namespace) -> None:
    profile = args.profile or resolve_profile(None)
    _ = _load_and_validate(profile)
    script = Path(__file__).resolve().parents[2] / "scripts" / "bench_generate.py"
    if not script.exists():
        print({"ok": False, "error": "bench_generate.py not found"})
        return
    cmd = [sys.executable, str(script), "--profile", profile, "--max-new-tokens", str(args.max_new_tokens)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="c3rnt2")
    sub = parser.add_subparsers(dest="command")

    doc = sub.add_parser("doctor")
    doc.add_argument("--profile", default=None)
    doc.add_argument("--deep", action="store_true")
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

    bench = sub.add_parser("bench")
    bench.add_argument("--profile", default=None)
    bench.add_argument("--max-new-tokens", type=int, default=512)
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
