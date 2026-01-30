from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import validate_profile
from .device import detect_device
from .model.core_transformer import CoreTransformer
from .model_loader import load_inference_model


def check_deps(modules: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for name in modules:
        try:
            __import__(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"missing ({exc.__class__.__name__})"
    return status


def run_deep_checks(settings: dict, base_dir: Path) -> dict[str, Any]:
    info = detect_device()
    if info.cuda_available:
        torch.cuda.reset_peak_memory_stats()
    local_settings = deepcopy(settings)
    core_cfg = dict(local_settings.get("core", {}) or {})
    if core_cfg.get("cuda_graphs"):
        core_cfg["cuda_graphs"] = False
        local_settings["core"] = core_cfg
    if core_cfg.get("compile") or core_cfg.get("compile_step") or core_cfg.get("compile_local_mixer_step"):
        try:
            import triton  # type: ignore
        except Exception:
            core_cfg["compile"] = False
            core_cfg["compile_step"] = False
            core_cfg["compile_local_mixer_step"] = False
            local_settings["core"] = core_cfg

    try:
        model = load_inference_model(local_settings)
    except Exception as exc:
        msg = str(exc).lower()
        if "triton" in msg or "inductor" in msg:
            core = dict(local_settings.get("core", {}) or {})
            if core.get("compile") or core.get("compile_step") or core.get("compile_local_mixer_step"):
                core["compile"] = False
                core["compile_step"] = False
                core["compile_local_mixer_step"] = False
                local_settings["core"] = core
                model = load_inference_model(local_settings)
            else:
                raise
        else:
            raise
    prompt = "def add(a, b):"
    ids, _ = model.encode_prompt(prompt)
    if not ids:
        ids = [0]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    start = time.time()
    with torch.inference_mode():
        if hasattr(model, "step_block"):
            _ = model.forward(input_ids)
            state = model.init_state(prompt_ids=ids)
            _ = model.step(ids[-1], state)
            block_ids = torch.tensor([ids[: max(1, min(2, len(ids)))]], dtype=torch.long, device=model.device)
            _ = model.step_block(block_ids, state)
            last_tok = ids[-1]
            gen_state = state
            gen_tokens = 16
            gen_start = time.time()
            for _ in range(gen_tokens):
                logits, gen_state = model.step(last_tok, gen_state)
                last_tok = int(torch.argmax(logits, dim=-1).item())
            gen_elapsed = max(1e-6, time.time() - gen_start)
            tokens_per_sec = gen_tokens / gen_elapsed
        else:
            gen_tokens = 16
            gen_start = time.time()
            _ = model.generate(prompt, max_new_tokens=gen_tokens)
            gen_elapsed = max(1e-6, time.time() - gen_start)
            tokens_per_sec = gen_tokens / gen_elapsed
    elapsed = time.time() - start
    vram_peak_gb = None
    if info.cuda_available:
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return {
        "deep_ok": True,
        "elapsed_sec": round(elapsed, 3),
        "vram_peak_gb": round(vram_peak_gb, 3) if vram_peak_gb is not None else None,
        "tokens_per_sec": round(tokens_per_sec, 3),
    }


def doctor_report(settings: dict, base_dir: Path, deep: bool = False) -> dict[str, Any]:
    info = detect_device()
    report: Dict[str, Any] = {
        "device": info.device,
        "cuda_available": info.cuda_available,
        "gpu": info.name,
        "vram_gb": info.vram_gb,
        "dtype": info.dtype,
        "python": sys.version.split()[0],
    }
    validate_profile(settings, base_dir=base_dir)
    if deep:
        report.update(run_deep_checks(settings, base_dir=base_dir))
    return report
