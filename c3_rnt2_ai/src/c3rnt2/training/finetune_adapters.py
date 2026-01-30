from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ..config import load_settings, validate_profile
from .lora_utils import load_samples_from_path, hash_files, train_lora


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def finetune_adapter(
    settings: dict,
    adapter_path: Path,
    data_path: Path,
    output_path: Path,
    steps: int,
    lr: float,
    batch_tokens: int,
    grad_accum: int,
) -> dict:
    adapter_cfg = settings.get("continuous", {}).get("adapters", {}) or {}
    samples = load_samples_from_path(data_path, source_kind="finetune")
    if not samples:
        raise ValueError("No samples provided")
    stats = train_lora(
        settings,
        samples,
        steps=steps,
        lr=lr,
        batch_tokens=batch_tokens,
        grad_accum=grad_accum,
        adapter_path=output_path,
        adapter_cfg=adapter_cfg,
        init_adapter_path=adapter_path,
    )
    parent_hash = _hash_file(adapter_path)
    manifest = {
        "adapter_path": str(output_path),
        "parent_adapter": str(adapter_path),
        "parent_hash": parent_hash,
        "rank": int(adapter_cfg.get("rank", 4)),
        "alpha": float(adapter_cfg.get("alpha", 1.0)),
        "targets": adapter_cfg.get("target_modules"),
        "steps": steps,
        "lr": lr,
        "batch_tokens": batch_tokens,
        "grad_accum": grad_accum,
        "dataset_hash": hash_files([data_path]) if data_path.is_file() else hash_files(list(data_path.rglob("*"))),
        "ts": time.time(),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    return {"ok": True, "adapter": str(output_path), "manifest": str(manifest_path), "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/experts/finetuned.pt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings(args.profile)
    validate_profile(settings, base_dir=Path("."))
    result = finetune_adapter(
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


if __name__ == "__main__":
    main()
