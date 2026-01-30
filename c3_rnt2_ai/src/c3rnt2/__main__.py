from __future__ import annotations

import argparse
import sys
import time
import subprocess

try:
    import torch
except Exception:  # pragma: no cover
    torch = None
from copy import deepcopy
from pathlib import Path

from .config import load_settings, resolve_profile, validate_profile
from .doctor import check_deps, run_deep_checks
from .model.core_transformer import CoreTransformer, save_checkpoint
from .model_loader import load_inference_model
from .training import eval as eval_mod
from .tokenizer import rnt2_train
from .agent.agent_loop import run_demo_agent
from .continuous.trainer import ContinualTrainer
from .continuous.lora import load_lora_state, inject_lora, LoRAConfig, resolve_target_modules
from .continuous.dataset import retrieve_context, ingest_sources, collect_samples
from .continuous.anchors import write_default_anchors
from .continuous.registry import load_registry, is_bootstrapped
from .prompting.chat_format import build_chat_prompt
from .continuous.bootstrap import run_bootstrap
from .device import detect_device
from .selfimprove.improve_loop import run_improve_loop
from .selfimprove.patch_ops import apply_patch
from .utils.locks import acquire_exclusive_lock, LockUnavailable
from .runtime.router import build_features, load_router, log_router_event
from .training import train_router as train_router_mod
from .training import train_experts as train_experts_mod
from .training import finetune_adapters as finetune_mod


def _parse_interval(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    return int(interval)




def _load_and_validate(profile: str | None) -> dict:
    settings = load_settings(profile)
    validate_profile(settings, base_dir=Path('.'))
    return settings

def _default_profile() -> str:
    for candidate in ("rtx4080_16gb_vortexx_next", "dev_small"):
        try:
            load_settings(candidate)
            return candidate
        except Exception:
            continue
    return resolve_profile(None)


def cmd_tokenizer_train(args: argparse.Namespace) -> None:
    sub_block_sizes = [int(x) for x in args.sub_block_sizes.split(",")] if args.sub_block_sizes else None
    sub_codebook_sizes = [int(x) for x in args.sub_codebook_sizes.split(",")] if args.sub_codebook_sizes else None
    rnt2_train.train(
        args.codebook_size,
        args.block_size,
        Path(args.corpus),
        Path(args.output),
        vortex_output=Path(args.vortex_output),
        macro_size=args.macro_size,
        macro_min_len=args.macro_min_len,
        sub_block_size=args.sub_block_size,
        sub_codebook_size=args.sub_codebook_size,
        sub_block_sizes=sub_block_sizes,
        sub_codebook_sizes=sub_codebook_sizes,
    )


def cmd_eval(_args: argparse.Namespace) -> None:
    eval_mod.main()


def cmd_chat(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    model = load_inference_model(settings)
    # Load current adapter if available (core backend only)
    if hasattr(model, "blocks"):
        state = load_registry(Path("."))
        if state.current_run_id:
            adapter_path = Path("data") / "registry" / "adapters" / f"{state.current_run_id}.pt"
            if adapter_path.exists():
                adapter_cfg = settings.get("continuous", {}).get("adapters", {})
                lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
                strict = bool(adapter_cfg.get("strict_target_modules", False))
                target_modules = resolve_target_modules(adapter_cfg, strict=strict)
                inject_lora(model, lora_cfg, target_modules=target_modules)
                load_lora_state(model, adapter_path)
    router_cfg = settings.get("core", {}).get("router", settings.get("router", {})) or {}
    router = None
    if bool(router_cfg.get("enabled", False)):
        router_path = Path(router_cfg.get("path", "data/runs/router.pt"))
        router = load_router(router_path, router_path.with_suffix(".json"))
    info = detect_device()
    print({"device": info.device, "vram_gb": info.vram_gb, "dtype": info.dtype})
    decode_cfg = settings.get("decode", {})
    print("VORTEX-X chat. Type 'exit' to quit.")

    def _set_stream_topk(enable: bool) -> object | None:
        runtime = getattr(model, "runtime_cfg", None)
        if runtime is None:
            return None
        prev = runtime.get("paged_lm_head_stream_topk", False)
        if enable:
            runtime["paged_lm_head_stream_topk"] = int(router_cfg.get("stream_topk_k", 64))
        else:
            runtime["paged_lm_head_stream_topk"] = False
        return prev

    while True:
        prompt = input("> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        rag_cfg = settings.get("rag", {})
        if bool(rag_cfg.get("enabled", False)):
            top_k = int(rag_cfg.get("top_k", 3))
            context = retrieve_context(Path("."), prompt, settings, top_k=top_k)
            if context:
                prompt = f"Context:
{context}

User:
{prompt}"
        backend = settings.get("core", {}).get("backend", "vortex")
        default_system = settings.get("core", {}).get("hf_system_prompt", "You are a helpful coding assistant.")
        messages = [{"role": "user", "content": prompt}]
        prompt_text = build_chat_prompt(messages, backend, tokenizer=getattr(model, "tokenizer", None), default_system=default_system)
        max_new = args.max_new_tokens or int(decode_cfg.get("max_new_tokens", 64))
        decision = None
        prev_stream = None
        if router is not None:
            feats = build_features(prompt_text, max_new, settings)
            decision = router.decide(feats)
            prev_stream = _set_stream_topk(bool(decision.stream_topk))
        start = time.time()
        if hasattr(model, "blocks"):
            response, stats = model.generate(
                prompt_text,
                max_new_tokens=max_new,
                temperature=args.temperature if args.temperature is not None else float(decode_cfg.get("temperature", 1.0)),
                top_p=args.top_p if args.top_p is not None else float(decode_cfg.get("top_p", 1.0)),
                repetition_penalty=args.repetition_penalty if args.repetition_penalty is not None else float(decode_cfg.get("repetition_penalty", 1.0)),
                no_repeat_ngram=args.no_repeat_ngram if args.no_repeat_ngram is not None else int(decode_cfg.get("no_repeat_ngram", 0)),
                adaptive_granularity=args.adaptive_granularity or bool(decode_cfg.get("adaptive_granularity", False)),
                exact_copy_mode=bool(decode_cfg.get("exact_copy_mode", False)),
                escape_restrict=bool(decode_cfg.get("escape_restrict", False)),
                use_mtp=bool(decode_cfg.get("use_mtp", True)),
                return_stats=True,
            )
        else:
            response = model.generate(
                prompt_text,
                max_new_tokens=max_new,
                temperature=args.temperature if args.temperature is not None else float(decode_cfg.get("temperature", 1.0)),
                top_p=args.top_p if args.top_p is not None else float(decode_cfg.get("top_p", 1.0)),
                repetition_penalty=args.repetition_penalty if args.repetition_penalty is not None else float(decode_cfg.get("repetition_penalty", 1.0)),
                no_repeat_ngram=args.no_repeat_ngram if args.no_repeat_ngram is not None else int(decode_cfg.get("no_repeat_ngram", 0)),
                adaptive_granularity=args.adaptive_granularity or bool(decode_cfg.get("adaptive_granularity", False)),
                exact_copy_mode=bool(decode_cfg.get("exact_copy_mode", False)),
                escape_restrict=bool(decode_cfg.get("escape_restrict", False)),
                use_mtp=bool(decode_cfg.get("use_mtp", True)),
            )
            stats = None
        elapsed = max(1e-6, time.time() - start)
        vram_peak = None
        if torch is not None and torch.cuda.is_available():
            try:
                vram_peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            except Exception:
                vram_peak = None
        mem_cost = 0.0
        if hasattr(model, "blocks"):
            try:
                mem_cost = float(sum(block.lava.stats.reads + block.lava.stats.writes for block in model.blocks))
            except Exception:
                mem_cost = 0.0
        if router is not None:
            verify_rate = 0.0
            if stats is not None:
                verify_rate = float(stats.accepted) / max(1, int(stats.proposed))
            log_router_event(
                Path("."),
                {
                    "request_id": f"cli-{int(time.time())}",
                    "backend": "core" if backend != "hf" else "hf",
                    "stream_topk": bool(decision.stream_topk) if decision else False,
                    "prompt_tokens": len(prompt_text.split()),
                    "max_new_tokens": max_new,
                    "tok_per_s": float(max_new) / elapsed,
                    "verify_accept_rate": verify_rate,
                    "error_rate": 0.0,
                    "vram_peak_mb": vram_peak or 0.0,
                    "mem_cost_estimate": mem_cost,
                },
            )
        if prev_stream is not None:
            _set_stream_topk(bool(prev_stream))
        print(response)

def cmd_agent_demo(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    report = run_demo_agent(settings)
    print(report)


def cmd_self_train(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    info = detect_device()
    print({"device": info.device, "vram_gb": info.vram_gb, "dtype": info.dtype})
    base_dir = Path(".")
    if not is_bootstrapped(base_dir, settings):
        print({"ok": False, "error": "model not bootstrapped"})
        return
    try:
        lock = acquire_exclusive_lock(base_dir, "train")
    except LockUnavailable:
        print({"ok": False, "error": "train lock unavailable (serve running?)"})
        return
    try:
        trainer = ContinualTrainer(settings=settings, base_dir=base_dir)
        interval_sec = _parse_interval(args.interval)
        while True:
            result = trainer.run_tick(ingest=True)
            print({"run_id": result.run_id, "promoted": result.promoted, "loss": result.loss})
            if args.once:
                break
            time.sleep(interval_sec)
    finally:
        lock.release()


def cmd_self_improve(_args: argparse.Namespace) -> None:
    report = run_improve_loop(Path("."))
    print(report)


def cmd_apply_patch(args: argparse.Namespace) -> None:
    diff = Path(args.diff).read_text(encoding="utf-8")
    result = apply_patch(Path("."), diff, approve=args.approve)
    print({"ok": result.ok, "message": result.message})


def cmd_doctor(args: argparse.Namespace) -> None:
    info = detect_device()
    print({
        "device": info.device,
        "cuda_available": info.cuda_available,
        "gpu": info.name,
        "vram_gb": info.vram_gb,
        "dtype": info.dtype,
        "python": sys.version.split()[0],
    })
    modules = [
        "torch",
        "bitsandbytes",
        "faiss",
        "triton",
        "fastapi",
        "zstandard",
        "lz4",
    ]
    print({"deps": check_deps(modules)})
    base_dir = Path(".")
    try:
        settings = _load_and_validate(args.profile)
    except Exception as exc:
        print({"warning": "settings_invalid", "error": str(exc), "hint": "Update config/settings.yaml to include missing keys"})
        return
    print({"settings_ok": True, "profile": args.profile or resolve_profile(None)})
    if args.deep:
        try:
            deep_result = run_deep_checks(settings, base_dir=base_dir)
            print({"deep": deep_result})
        except Exception as exc:
            print({"deep": {"deep_ok": False, "error": str(exc)}})


def cmd_serve(args: argparse.Namespace) -> None:
    profile = args.profile or _default_profile()
    settings = _load_and_validate(profile)
    base_dir = Path(".")
    try:
        lock = acquire_exclusive_lock(base_dir, "serve")
    except LockUnavailable:
        print({"ok": False, "error": "serve lock unavailable (training running?)"})
        return
    try:
        from .server import run_server
    except Exception as exc:
        lock.release()
        print({"ok": False, "error": f"fastapi/uvicorn not available: {exc}"})
        return
    try:
        run_server(settings=settings, base_dir=base_dir, host=args.host, port=args.port)
    finally:
        lock.release()



def cmd_load_checkpoint(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    core = settings.get("core", {})
    core["checkpoint_path"] = args.path
    settings["core"] = core
    model = CoreTransformer.from_settings(settings)
    meta = getattr(model, "checkpoint_meta", {})
    print({"ok": True, "path": args.path, "meta": meta})


def cmd_save_checkpoint(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    model = CoreTransformer.from_settings(settings)
    save_checkpoint(model, Path(args.out), settings)
    print({"ok": True, "path": args.out})


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


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    allowlist = settings.get("agent", {}).get("web_allowlist", [])
    count = ingest_sources(Path("."), allowlist, settings)
    print({"ingested_docs": count})


def cmd_ingest_once(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    settings = deepcopy(settings)
    cont = settings.get("continuous", {}) or {}
    replay = cont.get("replay", {}) or {}
    replay["sample_size"] = 0
    cont["replay"] = replay
    settings["continuous"] = cont
    allowlist = settings.get("agent", {}).get("web_allowlist", [])
    collected = collect_samples(Path("."), allowlist, settings, ingest=True)
    print({"ingested_docs": collected.stats.new_docs, "filtered": collected.stats.filtered})


def cmd_train_once(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    base_dir = Path(".")
    if not is_bootstrapped(base_dir, settings):
        print({"ok": False, "error": "model not bootstrapped"})
        return
    try:
        lock = acquire_exclusive_lock(base_dir, "train")
    except LockUnavailable:
        print({"ok": False, "error": "train lock unavailable (serve running?)"})
        return
    try:
        trainer = ContinualTrainer(settings=settings, base_dir=base_dir)
        result = trainer.run_tick(ingest=False)
        print({"run_id": result.run_id, "promoted": result.promoted, "loss": result.loss})
    finally:
        lock.release()



def cmd_train_router(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    weights_cfg = settings.get("router", {}).get("weights", {})
    weights = {
        "speed": float(args.w_speed if args.w_speed is not None else weights_cfg.get("speed", 1.0)),
        "vram": float(args.w_vram if args.w_vram is not None else weights_cfg.get("vram", 0.01)),
        "error": float(args.w_error if args.w_error is not None else weights_cfg.get("error", 5.0)),
        "quality": float(args.w_quality if args.w_quality is not None else weights_cfg.get("quality", 1.0)),
    }
    result = train_router_mod.train_router(
        Path(args.data),
        Path(args.output),
        weights=weights,
        epochs=int(args.epochs),
        hidden_size=int(args.hidden),
        seed=int(args.seed),
    )
    print(result)


def cmd_train_experts(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        domains = [p.name for p in Path(args.data).iterdir() if p.is_dir()]
    result = train_experts_mod.train_experts(
        settings=settings,
        domains=domains,
        data_root=Path(args.data),
        output_root=Path(args.output),
        steps=int(args.steps),
        lr=float(args.lr),
        batch_tokens=int(args.batch_tokens),
        grad_accum=int(args.grad_accum),
    )
    print(result)


def cmd_finetune_adapter(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    result = finetune_mod.finetune_adapter(
        settings=settings,
        adapter_path=Path(args.adapter),
        data_path=Path(args.data),
        output_path=Path(args.output),
        steps=int(args.steps),
        lr=float(args.lr),
        batch_tokens=int(args.batch_tokens),
        grad_accum=int(args.grad_accum),
    )
    print(result)


def cmd_bench(args: argparse.Namespace) -> None:
    profile = args.profile or _default_profile()
    _ = _load_and_validate(profile)
    script = Path(__file__).resolve().parents[2] / "scripts" / "bench_generate.py"
    cmd = [sys.executable, str(script), "--profile", profile, "--max-new-tokens", str(args.max_new_tokens)]
    if args.bench_topk:
        cmd.append("--bench-topk")
    if args.use_cuda_graphs:
        cmd.append("--use-cuda-graphs")
    if args.use_compile:
        cmd.append("--use-compile")
    if args.use_full_profile:
        cmd.append("--use-full-profile")
    subprocess.run(cmd, check=True)


def cmd_anchors_init(args: argparse.Namespace) -> None:
    settings = _load_and_validate(args.profile)
    anchors_path = Path(settings.get("continuous", {}).get("eval", {}).get("anchors_path", "data/continuous/anchors.jsonl"))
    write_default_anchors(anchors_path)
    print({"anchors_path": str(anchors_path)})


def main() -> None:
    parser = argparse.ArgumentParser(prog="c3rnt2")
    sub = parser.add_subparsers(dest="command")

    tok = sub.add_parser("tokenizer-train")
    tok.add_argument("--corpus", default="data/corpora")
    tok.add_argument("--output", default="data/runs/rnt2_dev.pt")
    tok.add_argument("--vortex-output", default="data/runs/vortex_tok.pt")
    tok.add_argument("--block-size", type=int, default=64)
    tok.add_argument("--codebook-size", type=int, default=1024)
    tok.add_argument("--macro-size", type=int, default=256)
    tok.add_argument("--macro-min-len", type=int, default=2)
    tok.add_argument("--sub-block-size", type=int, default=16)
    tok.add_argument("--sub-codebook-size", type=int, default=256)
    tok.add_argument("--sub-block-sizes", type=str, default=None)
    tok.add_argument("--sub-codebook-sizes", type=str, default=None)
    tok.set_defaults(func=cmd_tokenizer_train)

    ev = sub.add_parser("eval")
    ev.set_defaults(func=cmd_eval)

    doc = sub.add_parser("doctor")
    doc.add_argument("--profile", default=None)
    doc.add_argument("--deep", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    chat = sub.add_parser("chat")
    chat.add_argument("--profile", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=None)
    chat.add_argument("--temperature", type=float, default=None)
    chat.add_argument("--top-p", type=float, default=None)
    chat.add_argument("--repetition-penalty", type=float, default=None)
    chat.add_argument("--no-repeat-ngram", type=int, default=None)
    chat.add_argument("--adaptive-granularity", action="store_true")
    chat.set_defaults(func=cmd_chat)

    agent = sub.add_parser("agent-demo")
    agent.add_argument("--profile", default=None)
    agent.set_defaults(func=cmd_agent_demo)

    serve = sub.add_parser("serve")
    serve.add_argument("--profile", default=None)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    st = sub.add_parser("self-train")
    st.add_argument("--profile", default=None)
    st.add_argument("--interval", default="30m")
    st.add_argument("--once", action="store_true")
    st.set_defaults(func=cmd_self_train)

    ing = sub.add_parser("ingest")
    ing.add_argument("--profile", default=None)
    ing.set_defaults(func=cmd_ingest)

    ing_once = sub.add_parser("ingest-once")
    ing_once.add_argument("--profile", default=None)
    ing_once.set_defaults(func=cmd_ingest_once)

    tr_once = sub.add_parser("train-once")
    tr_once.add_argument("--profile", default=None)
    tr_once.set_defaults(func=cmd_train_once)

    si = sub.add_parser("self-improve")
    si.set_defaults(func=cmd_self_improve)

    ap = sub.add_parser("apply-patch")
    ap.add_argument("--diff", required=True)
    ap.add_argument("--approve", action="store_true")
    ap.set_defaults(func=cmd_apply_patch)

    lc = sub.add_parser("load-checkpoint")
    lc.add_argument("--path", required=True)
    lc.add_argument("--profile", default=None)
    lc.set_defaults(func=cmd_load_checkpoint)

    sc = sub.add_parser("save-checkpoint")
    sc.add_argument("--out", required=True)
    sc.add_argument("--profile", default=None)
    sc.set_defaults(func=cmd_save_checkpoint)

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

    tr_router = sub.add_parser("train-router")
    tr_router.add_argument("--profile", default=None)
    tr_router.add_argument("--data", default="data/logs")
    tr_router.add_argument("--output", default="data/runs/router.pt")
    tr_router.add_argument("--epochs", type=int, default=200)
    tr_router.add_argument("--hidden", type=int, default=16)
    tr_router.add_argument("--seed", type=int, default=42)
    tr_router.add_argument("--w-speed", type=float, default=None)
    tr_router.add_argument("--w-vram", type=float, default=None)
    tr_router.add_argument("--w-error", type=float, default=None)
    tr_router.add_argument("--w-quality", type=float, default=None)
    tr_router.set_defaults(func=cmd_train_router)

    tr_exp = sub.add_parser("train-experts")
    tr_exp.add_argument("--profile", default=None)
    tr_exp.add_argument("--data", default="data/corpora")
    tr_exp.add_argument("--output", default="data/experts")
    tr_exp.add_argument("--domains", default="")
    tr_exp.add_argument("--steps", type=int, default=50)
    tr_exp.add_argument("--lr", type=float, default=1e-4)
    tr_exp.add_argument("--batch-tokens", type=int, default=2048)
    tr_exp.add_argument("--grad-accum", type=int, default=1)
    tr_exp.set_defaults(func=cmd_train_experts)

    ft_ad = sub.add_parser("finetune-adapter")
    ft_ad.add_argument("--profile", default=None)
    ft_ad.add_argument("--adapter", required=True)
    ft_ad.add_argument("--data", required=True)
    ft_ad.add_argument("--output", default="data/experts/finetuned.pt")
    ft_ad.add_argument("--steps", type=int, default=50)
    ft_ad.add_argument("--lr", type=float, default=1e-4)
    ft_ad.add_argument("--batch-tokens", type=int, default=2048)
    ft_ad.add_argument("--grad-accum", type=int, default=1)
    ft_ad.set_defaults(func=cmd_finetune_adapter)

    bench = sub.add_parser("bench")
    bench.add_argument("--profile", default=None)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--bench-topk", action="store_true")
    bench.add_argument("--use-cuda-graphs", action="store_true")
    bench.add_argument("--use-compile", action="store_true")
    bench.add_argument("--use-full-profile", action="store_true")
    bench.set_defaults(func=cmd_bench)

    anch = sub.add_parser("anchors-init")
    anch.add_argument("--profile", default=None)
    anch.set_defaults(func=cmd_anchors_init)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

