from __future__ import annotations

import gc
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from ..device import autocast_context
from ..model.core_transformer import CoreTransformer
from ..prompting.chat_format import build_chat_prompt
from .anchors import load_anchors
from .formatting import format_chat_sample
from .lora import LoRAConfig, inject_lora, resolve_target_modules, save_lora_state
from .registry import load_registry, save_registry, mark_bootstrapped
from .types import Sample


def _default_prompts() -> list[str]:
    return [
        "Explain how a hash map works and give a Python example.",
        "Write a Python function to reverse a linked list.",
        "Fix the bug in this code: for i in range(5): print(i) # should print 0..4",
        "Write tests for a function that parses JSON safely.",
        "Explain time complexity of quicksort vs mergesort.",
        "Complete the code: def is_prime(n): ...",
        "Detect a bug and propose a fix: list.remove used in a loop.",
        "Write a small HTTP client in Python using requests.",
        "Explain the difference between threads and async in Python.",
        "Write a SQL query to find top 5 users by spend.",
        "Describe how to avoid off-by-one errors in loops.",
        "Write a function to flatten a nested list.",
        "Complete this code snippet: class LRUCache: ...",
        "Explain why mutable default args are dangerous.",
        "Write tests for an LRU cache implementation.",
        "Find a bug: reading file without closing. Propose fix.",
        "Write a function to parse CSV with quoted fields.",
        "Explain how to handle Unicode in Python strings.",
        "Draft a docstring for a function that validates email.",
        "Write a function to compute factorial iteratively.",
        "Explain the difference between shallow and deep copy.",
        "Complete: def binary_search(arr, target): ...",
        "Explain REST vs RPC with a brief example.",
        "Write a pytest for a function that normalizes whitespace.",
        "Find bug: integer division in Python 3. Fix it.",
        "Write a function to merge two sorted lists.",
        "Explain why recursion can cause stack overflow.",
        "Complete a decorator that logs function calls.",
        "Detect bug: modifying dict while iterating. Fix it.",
        "Write a small CLI parser with argparse.",
        "Explain how to avoid SQL injection.",
        "Write tests for a CSV parser edge cases.",
        "Complete code: def parse_config(text): ...",
        "Explain big-O for dictionary lookup.",
        "Write a function to compute rolling average.",
        "Write unit tests for a cache eviction policy.",
        "Fix bug: string concatenation in a loop (performance).",
        "Explain how to design idempotent APIs.",
        "Complete: def top_k(nums, k): ...",
        "Write a function to validate parentheses in a string.",
        "Explain what a race condition is and an example.",
        "Write a test that checks exceptions are raised.",
        "Fix bug: using is instead of == for strings.",
        "Complete: class Stack: push/pop/is_empty",
        "Write tests for a function that parses dates.",
        "Explain what a context manager does in Python.",
        "Write a function to compute edit distance.",
        "Find bug: forgetting to return value. Fix it.",
    ]


def _load_prompts(settings: dict, max_prompts: int) -> list[str]:
    anchors_path = Path(settings.get("continuous", {}).get("eval", {}).get("anchors_path", "data/continuous/anchors.jsonl"))
    anchors = load_anchors(anchors_path)
    prompts = [a.prompt for a in anchors if a.prompt]
    if not prompts:
        prompts = _default_prompts()
    return prompts[: max(1, max_prompts)]


def _filter_response(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) < 40:
        return False
    words = cleaned.split()
    if not words:
        return False
    uniq = len(set(words)) / max(1, len(words))
    if uniq < 0.2:
        return False
    return True


def _bootstrap_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "registry" / "bootstrap"


def _dataset_path(base_dir: Path) -> Path:
    return _bootstrap_dir(base_dir) / "bootstrap_samples.jsonl"


def _save_dataset(path: Path, samples: List[Sample], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "prompt": sample.prompt,
                "response": sample.response,
                "teacher": meta.get("teacher"),
                "ts": meta.get("ts"),
                "settings_profile": meta.get("settings_profile"),
                "seed": meta.get("seed"),
                "params": meta.get("params", {}),
            }
            handle.write(json.dumps(record) + "\n")


def _load_dataset(path: Path) -> List[Sample]:
    if not path.exists():
        return []
    samples: List[Sample] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        prompt = str(payload.get("prompt", "")).strip()
        response = str(payload.get("response", "")).strip()
        if not prompt or not response:
            continue
        samples.append(Sample(prompt=prompt, response=response, source_kind="bootstrap"))
    return samples




def _resolve_teacher_input_device(teacher_model: Any, fallback: str) -> str:
    device_map = getattr(teacher_model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        first = next(iter(device_map.values()))
        if isinstance(first, int):
            return f"cuda:{first}"
        if isinstance(first, str):
            return first
    device_attr = getattr(teacher_model, "device", None)
    if device_attr is not None:
        try:
            return str(device_attr)
        except Exception:
            pass
    return fallback

def _parse_max_memory(spec: str | None) -> dict | None:
    if not spec:
        return None
    parsed: dict[str, str] = {}
    for part in str(spec).split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def _load_teacher(teacher: str, device: str, quant: str, max_memory: str | None) -> tuple[Any, Any, dict, str]:
    info: dict[str, Any] = {"quant": quant, "device": device}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers not available: {exc}")

    kwargs: dict[str, Any] = {}
    max_mem = _parse_max_memory(max_memory)
    if max_mem:
        kwargs["max_memory"] = max_mem

    use_quant = quant in {"4bit", "8bit"}
    if use_quant:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
            if quant == "4bit":
                kwargs["load_in_4bit"] = True
            else:
                kwargs["load_in_8bit"] = True
            if device == "cuda":
                if torch.cuda.device_count() == 1:
                    kwargs["device_map"] = {"": 0}
                else:
                    kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = "cpu"
        except Exception:
            info["quant_fallback"] = True
            use_quant = False
    info["use_quant"] = use_quant

    if not use_quant:
        dtype = torch.float16 if device == "cuda" else torch.float32
        kwargs["torch_dtype"] = dtype

    teacher_tok = AutoTokenizer.from_pretrained(teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher, **kwargs)
    if not use_quant:
        teacher_model.to(device)
    teacher_model.eval()

    input_device = _resolve_teacher_input_device(teacher_model, device)
    return teacher_model, teacher_tok, info, input_device


def _generate_teacher_samples(
    teacher: str,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
    quant: str,
    max_memory: str | None,
    default_system: str | None,
) -> tuple[List[Sample], dict]:
    teacher_model, teacher_tok, info, input_device = _load_teacher(teacher, device=device, quant=quant, max_memory=max_memory)
    samples: List[Sample] = []
    for prompt in prompts:
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = build_chat_prompt(messages, backend="hf", tokenizer=teacher_tok, default_system=default_system)
            inputs = teacher_tok(prompt_text, return_tensors="pt")
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = teacher_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                )
            gen_ids = output[0][inputs["input_ids"].shape[1] :]
            completion = teacher_tok.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            continue
        completion = completion.strip()
        if not _filter_response(completion):
            continue
        samples.append(Sample(prompt=prompt, response=completion, source_kind="bootstrap"))

    try:
        if not bool(info.get("use_quant")) and hasattr(teacher_model, "to"):
            teacher_model.to("cpu")
    except Exception:
        pass
    del teacher_model
    del teacher_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return samples, info


def _train_from_samples(
    settings: dict,
    samples: List[Sample],
    steps: int,
    batch_tokens: int,
    grad_accum_steps: int,
) -> tuple[CoreTransformer, float | None]:
    seed = settings.get("continuous", {}).get("seed", settings.get("seed"))
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    model = CoreTransformer.from_settings(settings)
    adapter_cfg = settings.get("continuous", {}).get("adapters", {})
    lora_cfg = LoRAConfig(
        rank=int(adapter_cfg.get("rank", settings.get("continuous", {}).get("adapter_rank", 4))),
        alpha=float(adapter_cfg.get("alpha", 1.0)),
    )
    strict = bool(adapter_cfg.get("strict_target_modules", False))
    target_modules = resolve_target_modules(adapter_cfg, strict=strict)
    inject_lora(model, lora_cfg, target_modules=target_modules)

    lr = float(settings.get("continuous", {}).get("lr", 1e-4))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    last_loss = None
    use_scaler = model.device.type == "cuda" and model.dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if steps <= 0:
        return model, last_loss

    step_idx = 0
    optimizer.zero_grad(set_to_none=True)
    for _ in range(max(1, steps)):
        sequences = []
        token_count = 0
        attempts = 0
        while token_count < batch_tokens and attempts < max(4, len(samples)):
            sample = random.choice(samples)
            text = format_chat_sample(sample)
            ids, _ = model.encode_prompt(text)
            attempts += 1
            if len(ids) < 2:
                continue
            seq = torch.tensor(ids, dtype=torch.long)
            sequences.append(seq)
            token_count += len(ids)
        if not sequences:
            continue
        inputs = [seq[:-1] for seq in sequences]
        targets = [seq[1:] for seq in sequences]
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=0).to(model.device)
        target_ids = pad_sequence(targets, batch_first=True, padding_value=-100).to(model.device)
        with autocast_context(enabled=model.device.type == "cuda", dtype=model.config.dtype):
            logits = model.forward(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
            loss = loss / max(1, grad_accum_steps)
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step_idx += 1
        if step_idx % grad_accum_steps == 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.item())
    if step_idx % grad_accum_steps != 0:
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return model, last_loss


def _distill_teacher(
    settings: dict,
    base_dir: Path,
    teacher: str,
    max_prompts: int,
    max_new_tokens: int,
    steps: int,
    teacher_device: str,
    teacher_quant: str,
    teacher_max_memory: str | None,
    reuse_dataset: bool,
    batch_tokens: int,
    grad_accum_steps: int,
    profile_name: str | None,
) -> dict[str, Any]:
    dataset_path = _dataset_path(base_dir)
    samples: List[Sample] = []
    if reuse_dataset and dataset_path.exists():
        samples = _load_dataset(dataset_path)
    else:
        prompts = _load_prompts(settings, max_prompts=max_prompts)
        try:
            samples, info = _generate_teacher_samples(
                teacher=teacher,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                device=teacher_device,
                quant=teacher_quant,
                max_memory=teacher_max_memory,
                default_system=settings.get("core", {}).get("hf_system_prompt", "You are a helpful coding assistant."),
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        meta = {
            "teacher": teacher,
            "ts": time.time(),
            "settings_profile": profile_name,
            "seed": settings.get("continuous", {}).get("seed", settings.get("seed")),
            "params": {
                "max_prompts": max_prompts,
                "max_new_tokens": max_new_tokens,
                "teacher_device": teacher_device,
                "teacher_quant": teacher_quant,
                "teacher_max_memory": teacher_max_memory,
            },
        }
        _save_dataset(dataset_path, samples, meta)

    if not samples:
        return {"ok": False, "error": "no distillation samples produced"}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, last_loss = _train_from_samples(
        settings=settings,
        samples=samples,
        steps=steps,
        batch_tokens=batch_tokens,
        grad_accum_steps=grad_accum_steps,
    )

    adapter_path = base_dir / "data" / "registry" / "adapters" / "bootstrap.pt"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_state(model, adapter_path)
    run_dir = base_dir / "data" / "registry" / "runs" / "bootstrap"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "loss": last_loss,
        "samples": len(samples),
        "source": "teacher",
        "teacher": teacher,
        "dataset_path": str(dataset_path),
        "ts": time.time(),
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    state = load_registry(base_dir)
    if state.current_run_id:
        state.history.append(state.current_run_id)
    state.current_run_id = "bootstrap"
    save_registry(base_dir, state)
    mark_bootstrapped(base_dir, {
        "source": "teacher",
        "teacher": teacher,
        "dataset_path": str(dataset_path),
        "samples": len(samples),
        "loss": last_loss,
        "quant": teacher_quant,
    })
    return {"ok": True, "mode": "distill", "samples": len(samples), "loss": last_loss, "dataset": str(dataset_path)}


def run_bootstrap(
    settings: dict,
    base_dir: Path,
    checkpoint: str | None = None,
    teacher: str | None = None,
    max_prompts: int = 16,
    max_new_tokens: int = 64,
    steps: int = 50,
    teacher_device: str = "cuda",
    teacher_quant: str = "none",
    teacher_max_memory: str | None = None,
    reuse_dataset: bool = False,
    batch_tokens: int = 4096,
    grad_accum_steps: int = 1,
    profile_name: str | None = None,
) -> dict[str, Any]:
    if teacher:
        return _distill_teacher(
            settings=settings,
            base_dir=base_dir,
            teacher=teacher,
            max_prompts=max_prompts,
            max_new_tokens=max_new_tokens,
            steps=steps,
            teacher_device=teacher_device,
            teacher_quant=teacher_quant,
            teacher_max_memory=teacher_max_memory,
            reuse_dataset=reuse_dataset,
            batch_tokens=batch_tokens,
            grad_accum_steps=grad_accum_steps,
            profile_name=profile_name,
        )

    if reuse_dataset:
        dataset_path = _dataset_path(base_dir)
        samples = _load_dataset(dataset_path)
        if not samples:
            return {"ok": False, "error": f"bootstrap dataset not found: {dataset_path}"}
        model, last_loss = _train_from_samples(settings, samples, steps, batch_tokens, grad_accum_steps)
        adapter_path = base_dir / "data" / "registry" / "adapters" / "bootstrap.pt"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        save_lora_state(model, adapter_path)
        run_dir = base_dir / "data" / "registry" / "runs" / "bootstrap"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {"loss": last_loss, "samples": len(samples), "source": "dataset", "dataset_path": str(dataset_path), "ts": time.time()}
        (run_dir / "meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
        state = load_registry(base_dir)
        if state.current_run_id:
            state.history.append(state.current_run_id)
        state.current_run_id = "bootstrap"
        save_registry(base_dir, state)
        mark_bootstrapped(base_dir, {"source": "dataset", "dataset_path": str(dataset_path), "samples": len(samples), "loss": last_loss})
        return {"ok": True, "mode": "dataset", "samples": len(samples), "loss": last_loss, "dataset": str(dataset_path)}

    core = settings.get("core", {}) or {}
    ckpt = checkpoint or core.get("checkpoint_path")
    if ckpt:
        path = Path(ckpt)
        if not path.exists():
            return {"ok": False, "error": f"checkpoint not found: {path}"}
        local_settings = deepcopy(settings)
        core = dict(core)
        core["checkpoint_path"] = str(path)
        local_settings["core"] = core
        _ = CoreTransformer.from_settings(local_settings)
        mark_bootstrapped(base_dir, {"source": "checkpoint", "path": str(path)})
        return {"ok": True, "mode": "checkpoint", "path": str(path)}

    return {"ok": False, "error": "no checkpoint or teacher provided"}
