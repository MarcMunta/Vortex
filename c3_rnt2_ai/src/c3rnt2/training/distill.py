from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_settings, validate_profile
from ..continuous.bootstrap import run_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--teacher", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--teacher-device", type=str, default="cuda")
    parser.add_argument("--teacher-quant", type=str, default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--teacher-max-memory", type=str, default=None)
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings(args.profile)
    validate_profile(settings, base_dir=Path("."))
    result = run_bootstrap(
        settings=settings,
        base_dir=Path("."),
        checkpoint=args.checkpoint or None,
        teacher=args.teacher or None,
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


if __name__ == "__main__":
    main()
