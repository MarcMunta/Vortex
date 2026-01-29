from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_settings
from .model.core_transformer import CoreTransformer
from .training import eval as eval_mod
from .tokenizer import rnt2_train
from .agent.agent_loop import run_demo_agent
from .continuous.trainer import ContinualTrainer
from .continuous.lora import load_lora_state, inject_lora, LoRAConfig, resolve_target_modules
from .continuous.dataset import retrieve_context, ingest_sources
from .continuous.anchors import write_default_anchors
from .continuous.registry import load_registry
from .device import detect_device
from .selfimprove.improve_loop import run_improve_loop
from .selfimprove.patch_ops import apply_patch


def _parse_interval(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    return int(interval)


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
    settings = load_settings(args.profile)
    model = CoreTransformer.from_settings(settings)
    # Load current adapter if available
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
    info = detect_device()
    print({"device": info.device, "vram_gb": info.vram_gb, "dtype": info.dtype})
    decode_cfg = settings.get("decode", {})
    print("VORTEX-X chat. Type 'exit' to quit.")
    while True:
        prompt = input("> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        rag_cfg = settings.get("rag", {})
        if bool(rag_cfg.get("enabled", False)):
            top_k = int(rag_cfg.get("top_k", 3))
            context = retrieve_context(Path("."), prompt, settings, top_k=top_k)
            if context:
                prompt = f"Context:\n{context}\n\nUser:\n{prompt}"
        response = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens or int(decode_cfg.get("max_new_tokens", 64)),
            temperature=args.temperature if args.temperature is not None else float(decode_cfg.get("temperature", 1.0)),
            top_p=args.top_p if args.top_p is not None else float(decode_cfg.get("top_p", 1.0)),
            repetition_penalty=args.repetition_penalty if args.repetition_penalty is not None else float(decode_cfg.get("repetition_penalty", 1.0)),
            no_repeat_ngram=args.no_repeat_ngram if args.no_repeat_ngram is not None else int(decode_cfg.get("no_repeat_ngram", 0)),
            adaptive_granularity=args.adaptive_granularity or bool(decode_cfg.get("adaptive_granularity", False)),
            exact_copy_mode=bool(decode_cfg.get("exact_copy_mode", False)),
            escape_restrict=bool(decode_cfg.get("escape_restrict", False)),
            use_mtp=bool(decode_cfg.get("use_mtp", True)),
        )
        print(response)


def cmd_agent_demo(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
    report = run_demo_agent(settings)
    print(report)


def cmd_self_train(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
    info = detect_device()
    print({"device": info.device, "vram_gb": info.vram_gb, "dtype": info.dtype})
    trainer = ContinualTrainer(settings=settings, base_dir=Path("."))
    interval_sec = _parse_interval(args.interval)
    while True:
        result = trainer.run_tick()
        print({"run_id": result.run_id, "promoted": result.promoted, "loss": result.loss})
        if args.once:
            break
        time.sleep(interval_sec)


def cmd_self_improve(_args: argparse.Namespace) -> None:
    report = run_improve_loop(Path("."))
    print(report)


def cmd_apply_patch(args: argparse.Namespace) -> None:
    diff = Path(args.diff).read_text(encoding="utf-8")
    result = apply_patch(Path("."), diff, approve=args.approve)
    print({"ok": result.ok, "message": result.message})


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
    allowlist = settings.get("agent", {}).get("web_allowlist", [])
    count = ingest_sources(Path("."), allowlist, settings)
    print({"ingested_docs": count})


def cmd_anchors_init(args: argparse.Namespace) -> None:
    settings = load_settings(args.profile)
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

    st = sub.add_parser("self-train")
    st.add_argument("--profile", default=None)
    st.add_argument("--interval", default="30m")
    st.add_argument("--once", action="store_true")
    st.set_defaults(func=cmd_self_train)

    si = sub.add_parser("self-improve")
    si.set_defaults(func=cmd_self_improve)

    ap = sub.add_parser("apply-patch")
    ap.add_argument("--diff", required=True)
    ap.add_argument("--approve", action="store_true")
    ap.set_defaults(func=cmd_apply_patch)

    ing = sub.add_parser("ingest")
    ing.add_argument("--profile", default=None)
    ing.set_defaults(func=cmd_ingest)

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
