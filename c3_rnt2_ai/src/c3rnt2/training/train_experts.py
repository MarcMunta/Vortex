from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from ..config import load_settings, validate_profile
from ..continuous.anchors import load_anchors
from .lora_utils import load_samples_from_path, hash_files, eval_loss, train_lora


def _collect_domain_samples(data_root: Path, domains: List[str]) -> Dict[str, list]:
    samples_by_domain: Dict[str, list] = {}
    for domain in domains:
        domain_dir = data_root / domain
        if not domain_dir.exists():
            continue
        samples = load_samples_from_path(domain_dir, source_kind=domain)
        samples_by_domain[domain] = samples
    return samples_by_domain


def train_experts(
    settings: dict,
    domains: List[str],
    data_root: Path,
    output_root: Path,
    steps: int,
    lr: float,
    batch_tokens: int,
    grad_accum: int,
) -> Dict[str, object]:
    adapter_cfg = settings.get("continuous", {}).get("adapters", {}) or {}
    anchors_path = Path(settings.get("continuous", {}).get("eval", {}).get("anchors_path", "data/continuous/anchors.jsonl"))
    anchors = load_anchors(anchors_path)
    results: Dict[str, object] = {"domains": {}}

    samples_by_domain = _collect_domain_samples(data_root, domains)
    for domain, samples in samples_by_domain.items():
        if not samples:
            results["domains"][domain] = {"ok": False, "error": "no_samples"}
            continue
        adapter_path = output_root / domain / "adapter.pt"
        stats = train_lora(
            settings,
            samples,
            steps=steps,
            lr=lr,
            batch_tokens=batch_tokens,
            grad_accum=grad_accum,
            adapter_path=adapter_path,
            adapter_cfg=adapter_cfg,
        )
        # Evaluate anchors using a fresh model with the adapter loaded
        base_loss = None
        new_loss = None
        try:
            from ..model.core_transformer import CoreTransformer
            from ..continuous.lora import LoRAConfig, inject_lora, load_lora_state, resolve_target_modules
            import torch

            model = CoreTransformer.from_settings(settings)
            lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
            strict = bool(adapter_cfg.get("strict_target_modules", False))
            targets = resolve_target_modules(adapter_cfg, strict=strict)
            inject_lora(model, lora_cfg, target_modules=targets)
            backend = str(settings.get("core", {}).get("backend", "vortex"))
            default_system = settings.get("core", {}).get("hf_system_prompt", "You are Vortex, a helpful coding assistant.")
            tokenizer = getattr(model, "tokenizer", None)
            base_loss = eval_loss(model, anchors, backend=backend, tokenizer=tokenizer, default_system=default_system) if anchors else None
            load_lora_state(model, adapter_path)
            new_loss = eval_loss(model, anchors, backend=backend, tokenizer=tokenizer, default_system=default_system) if anchors else None
        except Exception:
            base_loss = None
            new_loss = None

        regression = 0.0
        if base_loss is not None and new_loss is not None:
            regression = (new_loss - base_loss) / max(1e-6, base_loss)
        max_regression = float(settings.get("continuous", {}).get("eval", {}).get("max_regression", 0.2))
        passed = regression <= max_regression
        manifest = {
            "domain": domain,
            "adapter_path": str(adapter_path),
            "rank": int(adapter_cfg.get("rank", 4)),
            "alpha": float(adapter_cfg.get("alpha", 1.0)),
            "targets": adapter_cfg.get("target_modules"),
            "steps": steps,
            "lr": lr,
            "batch_tokens": batch_tokens,
            "grad_accum": grad_accum,
            "dataset_hash": hash_files(list((data_root / domain).rglob("*"))),
            "anchor_base_loss": base_loss,
            "anchor_new_loss": new_loss,
            "anchor_regression": regression,
            "passed_eval": passed,
            "ts": time.time(),
        }
        manifest_path = output_root / domain / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
        results["domains"][domain] = {"ok": True, "adapter": str(adapter_path), "manifest": str(manifest_path), "stats": stats}

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/corpora")
    parser.add_argument("--output", type=str, default="data/experts")
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings(args.profile)
    validate_profile(settings, base_dir=Path("."))
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        domains = [p.name for p in Path(args.data).iterdir() if p.is_dir()]
    result = train_experts(
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


if __name__ == "__main__":
    main()
