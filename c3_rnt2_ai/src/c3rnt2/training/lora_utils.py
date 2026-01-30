from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence

from ..device import autocast_context
from ..model.core_transformer import CoreTransformer
from ..continuous.types import Sample
from ..continuous.formatting import format_chat_sample
from ..continuous.lora import LoRAConfig, LoRALinear, inject_lora, load_lora_state, save_lora_state, resolve_target_modules


def load_samples_from_path(path: Path, source_kind: str) -> List[Sample]:
    samples: List[Sample] = []
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_dir():
                continue
            if file.suffix.lower() == ".jsonl":
                for line in file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    prompt = str(payload.get("prompt", ""))
                    response = str(payload.get("response", payload.get("text", "")))
                    if not prompt and not response:
                        continue
                    samples.append(Sample(prompt=prompt, response=response, source_kind=source_kind))
            else:
                content = file.read_text(encoding="utf-8", errors="ignore").strip()
                if not content:
                    continue
                samples.append(Sample(prompt="", response=content, source_kind=source_kind))
    elif path.is_file():
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                prompt = str(payload.get("prompt", ""))
                response = str(payload.get("response", payload.get("text", "")))
                if not prompt and not response:
                    continue
                samples.append(Sample(prompt=prompt, response=response, source_kind=source_kind))
        else:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                samples.append(Sample(prompt="", response=content, source_kind=source_kind))
    return samples


def hash_files(paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def eval_loss(model: CoreTransformer, samples: List[Sample]) -> float | None:
    if not samples:
        return None
    model.eval()
    losses = []
    with torch.inference_mode():
        for sample in samples:
            text = format_chat_sample(sample)
            if not text:
                continue
            ids, _ = model.encode_prompt(text)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=model.device)
            targets = torch.tensor([ids[1:]], dtype=torch.long, device=model.device)
            with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
                logits = model.forward(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            losses.append(float(loss.item()))
    if not losses:
        return None
    return sum(losses) / max(1, len(losses))


def train_lora(
    settings: dict,
    samples: List[Sample],
    steps: int,
    lr: float,
    batch_tokens: int,
    grad_accum: int,
    adapter_path: Path,
    adapter_cfg: dict,
    init_adapter_path: Path | None = None,
) -> Dict[str, object]:
    if not samples:
        raise ValueError("No samples provided")
    model = CoreTransformer.from_settings(settings)
    strict = bool(adapter_cfg.get("strict_target_modules", False))
    target_modules = resolve_target_modules(adapter_cfg, strict=strict)
    lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
    inject_lora(model, lora_cfg, target_modules=target_modules)
    # Freeze base params; train only LoRA weights.
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
    # Disable LAVA writes during LoRA training to avoid inplace state updates.
    for block in getattr(model, "blocks", []):
        try:
            block.lava.enable_write = False
        except Exception:
            pass
    if init_adapter_path is not None:
        load_lora_state(model, init_adapter_path)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    use_scaler = model.device.type == "cuda" and model.dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    step_idx = 0
    tokens_seen = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step_idx < steps:
        sequences = []
        token_count = 0
        attempts = 0
        while token_count < batch_tokens and attempts < max(4, len(samples)):
            sample = samples[attempts % len(samples)]
            text = format_chat_sample(sample)
            ids, _ = model.encode_prompt(text)
            attempts += 1
            if len(ids) < 2:
                continue
            seq = torch.tensor(ids, dtype=torch.long)
            sequences.append(seq)
            token_count += len(ids)
        if not sequences:
            break
        inputs = [seq[:-1] for seq in sequences]
        targets = [seq[1:] for seq in sequences]
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=0).to(model.device)
        target_ids = pad_sequence(targets, batch_first=True, padding_value=-100).to(model.device)
        tokens_seen += int(target_ids.numel())
        with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
            logits = model.forward(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
            loss = loss / max(1, grad_accum)
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step_idx += 1
        if step_idx % grad_accum == 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if step_idx % grad_accum != 0:
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_state(model, adapter_path)
    elapsed = max(1e-6, time.time() - start)
    tps = tokens_seen / elapsed if tokens_seen else 0.0
    return {"steps": step_idx, "tokens": tokens_seen, "tokens_per_sec": tps, "adapter_path": str(adapter_path)}
