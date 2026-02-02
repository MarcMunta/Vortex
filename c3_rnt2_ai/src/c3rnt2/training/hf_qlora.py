from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..continuous.knowledge_store import KnowledgeStore, KnowledgeChunk, EmbeddingBackend
from ..continuous.types import Sample
from ..continuous.formatting import format_chat_sample
from ..continuous.anchors import load_anchors, write_default_anchors
from .dataset_builder import build_sft_dataset


@dataclass
class HfTrainResult:
    ok: bool
    run_id: str
    adapter_dir: Path | None
    loss: float | None
    steps: int
    samples: int
    tokens_per_sec: float | None
    error: str | None = None


class SFTDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: object, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate_fn(batch: List[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attention = []
    labels = []
    for item in batch:
        ids = item["input_ids"]
        pad = max_len - ids.size(0)
        if pad:
            ids = torch.nn.functional.pad(ids, (0, pad), value=pad_token_id)
        input_ids.append(ids)
        attn = item.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(item["input_ids"])
        if pad:
            attn = torch.nn.functional.pad(attn, (0, pad), value=0)
        attention.append(attn)
        lbl = item["labels"]
        if pad:
            lbl = torch.nn.functional.pad(lbl, (0, pad), value=-100)
        labels.append(lbl)
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def build_sft_texts(
    samples: Iterable[Sample],
    *,
    tokenizer: object | None,
    default_system: str | None,
) -> List[str]:
    texts: List[str] = []
    for sample in samples:
        text = format_chat_sample(sample, backend="hf", tokenizer=tokenizer, default_system=default_system)
        if text:
            texts.append(text)
    return texts


def build_sft_samples_from_chunks(
    chunks: List[KnowledgeChunk],
    prompt_template: str,
) -> List[Sample]:
    samples: List[Sample] = []
    for chunk in chunks:
        prompt = prompt_template.replace("{text}", chunk.text)
        samples.append(Sample(prompt=prompt, response=chunk.text, source_kind=chunk.source_kind))
    return samples


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state_path: Path, payload: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _resolve_registry_dir(base_dir: Path, settings: dict) -> Path:
    cfg = settings.get("hf_train", {}) or {}
    reg_dir = cfg.get("registry_dir", "data/registry/hf_train")
    reg_path = Path(reg_dir)
    if not reg_path.is_absolute():
        reg_path = base_dir / reg_path
    return reg_path


def _resolve_state_path(base_dir: Path, settings: dict) -> Path:
    cfg = settings.get("hf_train", {}) or {}
    state_path = cfg.get("state_path", "data/registry/hf_train/state.json")
    path = Path(state_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _resolve_dataset_path(base_dir: Path, settings: dict) -> Path:
    cfg = settings.get("hf_train", {}) or {}
    dataset_path = cfg.get("dataset_path", "data/registry/hf_train/sft_samples.jsonl")
    path = Path(dataset_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _save_dataset(path: Path, samples: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "prompt": sample.prompt,
                "response": sample.response,
                "source_kind": sample.source_kind,
                "messages": sample.messages,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_dataset(path: Path) -> List[Sample]:
    if not path.exists():
        return []
    samples: List[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        samples.append(
            Sample(
                prompt=str(payload.get("prompt", "")),
                response=str(payload.get("response", "")),
                source_kind=str(payload.get("source_kind", "unknown")),
                messages=payload.get("messages"),
            )
        )
    return samples


def _pack_texts(texts: List[str], tokenizer: object, max_tokens: int) -> List[str]:
    packed: List[str] = []
    current: List[str] = []
    current_len = 0
    for text in texts:
        try:
            length = len(tokenizer(text)["input_ids"])
        except Exception:
            length = len(text.split())
        if current and current_len + length > max_tokens:
            packed.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(text)
        current_len += length
    if current:
        packed.append("\n\n".join(current))
    return packed


def _bucket_by_length(texts: List[str], tokenizer: object) -> List[str]:
    lengths = []
    for text in texts:
        try:
            length = len(tokenizer(text)["input_ids"])
        except Exception:
            length = len(text.split())
        lengths.append((length, text))
    lengths.sort(key=lambda x: x[0])
    return [t for _, t in lengths]


def _eval_loss(model, tokenizer, samples: List[Sample], max_length: int) -> float:
    losses: List[float] = []
    with torch.inference_mode():
        for sample in samples:
            text = format_chat_sample(sample, backend="hf", tokenizer=tokenizer, default_system=None)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses.append(float(out.loss.item()))
    return sum(losses) / max(1, len(losses))


def _repeat_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    unique = len(set(words))
    return 1.0 - (unique / max(1, len(words)))


def train_once(settings: dict, base_dir: Path, reuse_dataset: bool = False) -> HfTrainResult:
    cfg = settings.get("hf_train", {}) or {}
    model_name = cfg.get("model_name") or settings.get("core", {}).get("hf_model")
    if not model_name:
        return HfTrainResult(ok=False, run_id="", adapter_dir=None, loss=None, steps=0, samples=0, tokens_per_sec=None, error="model_name missing")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel  # type: ignore
    except Exception as exc:
        return HfTrainResult(ok=False, run_id="", adapter_dir=None, loss=None, steps=0, samples=0, tokens_per_sec=None, error=f"hf deps missing: {exc}")

    reg_dir = _resolve_registry_dir(base_dir, settings)
    state_path = _resolve_state_path(base_dir, settings)
    dataset_path = _resolve_dataset_path(base_dir, settings)
    state = _load_state(state_path)
    last_ts = float(state.get("last_ts", 0.0))

    knowledge_path = Path(settings.get("continuous", {}).get("knowledge_path", base_dir / "data" / "continuous" / "knowledge.sqlite"))
    knowledge_cfg = settings.get("knowledge", {}) or {}
    embed_backend = knowledge_cfg.get("embedding_backend", "auto")
    embed_model = knowledge_cfg.get("embedding_model")
    embedder = EmbeddingBackend(backend=str(embed_backend), model_name=embed_model) if embed_model else embed_backend
    store = KnowledgeStore(
        knowledge_path,
        embedding_backend=embedder,
        index_backend=knowledge_cfg.get("index_backend", "auto"),
    )
    max_samples = int(cfg.get("max_samples", 128))
    min_quality = float(cfg.get("min_quality", 0.0))
    chunks = store.sample_chunks(limit=max_samples, min_quality=min_quality, since_ts=last_ts)
    episodes_path = base_dir / "data" / "episodes" / "agent.jsonl"
    if not chunks and not episodes_path.exists() and not reuse_dataset:
        return HfTrainResult(ok=False, run_id="", adapter_dir=None, loss=None, steps=0, samples=0, tokens_per_sec=None, error="no_samples")

    if reuse_dataset and dataset_path.exists():
        samples = _load_dataset(dataset_path)
    else:
        system_prompt = cfg.get("default_system") or settings.get("core", {}).get("hf_system_prompt") or "You are a helpful coding assistant."
        queue_dir = Path(settings.get("self_patch", {}).get("queue_dir", "data/self_patch/queue"))
        if not queue_dir.is_absolute():
            queue_dir = base_dir / queue_dir
        chat_path = base_dir / "data" / "episodes" / "chat.jsonl"
        feedback_path = base_dir / "data" / "episodes" / "feedback.jsonl"
        training_path = base_dir / "data" / "episodes" / "training.jsonl"
        build_sft_dataset(
            chunks=chunks,
            episodes_path=episodes_path,
            output_path=dataset_path,
            system_prompt=system_prompt,
            queue_dir=queue_dir,
            chat_path=chat_path,
            feedback_path=feedback_path,
            training_path=training_path,
            include_soft_feedback=bool(cfg.get("include_soft_feedback", True)),
            min_chars=int(cfg.get("min_chars", 40)),
            max_repeat_ratio=float(cfg.get("max_repeat_ratio", 0.8)),
            semantic_dedup_threshold=float(cfg.get("semantic_dedup_threshold", 0.97)),
            embedding_backend=embedder if isinstance(embedder, EmbeddingBackend) else None,
        )
        samples = _load_dataset(dataset_path)

    if not samples:
        return HfTrainResult(ok=False, run_id="", adapter_dir=None, loss=None, steps=0, samples=0, tokens_per_sec=None, error="empty_dataset")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    adapter_dir = reg_dir / run_id / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = cfg.get("compute_dtype", "bf16")
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    load_in_8bit = bool(cfg.get("load_in_8bit", False))
    quant_config = None
    if load_in_4bit or load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=bool(cfg.get("double_quant", True)),
            bnb_4bit_quant_type=str(cfg.get("quant_type", "nf4")),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=bool(cfg.get("use_fast", True)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = {"torch_dtype": torch_dtype}
    attn_impl = cfg.get("attn_implementation") or settings.get("core", {}).get("hf_attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    max_memory = cfg.get("max_memory")
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto" if device.startswith("cuda") else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    base_eval_samples: list[Sample] = []
    eval_cfg = cfg.get("eval", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    if eval_enabled:
        anchors_path = Path(settings.get("continuous", {}).get("eval", {}).get("anchors_path", base_dir / "data" / "continuous" / "anchors.jsonl"))
        if not anchors_path.is_absolute():
            anchors_path = base_dir / anchors_path
        if not anchors_path.exists():
            write_default_anchors(anchors_path)
        anchors = load_anchors(anchors_path)
        base_eval_samples = anchors[: int(eval_cfg.get("max_samples", 8) or 8)]
        if base_eval_samples:
            base_loss = _eval_loss(model, tokenizer, base_eval_samples, max_length=int(cfg.get("max_seq_len", 1024)))
        else:
            base_loss = None
    else:
        base_loss = None
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    adapter_path = cfg.get("adapter_path")
    if adapter_path:
        adapter_path = Path(adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = base_dir / adapter_path
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_rank", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=list(cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    max_length = int(cfg.get("max_seq_len", 1024))
    default_system = cfg.get("default_system") or settings.get("core", {}).get("hf_system_prompt")
    texts = build_sft_texts(samples, tokenizer=tokenizer, default_system=default_system)
    if bool(cfg.get("pack_samples", False)):
        texts = _pack_texts(texts, tokenizer, max_tokens=max_length)
    if bool(cfg.get("bucket_by_length", True)):
        texts = _bucket_by_length(texts, tokenizer)
    dataset = SFTDataset(texts, tokenizer=tokenizer, max_length=max_length)
    micro_batch = max(1, int(cfg.get("micro_batch_size", 1)))
    loader = DataLoader(dataset, batch_size=micro_batch, shuffle=True, collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id))

    lr = float(cfg.get("lr", 2e-4))
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if device.startswith("cuda"):
        try:
            optimizer = torch.optim.AdamW(optim_params, lr=lr, fused=True)
        except Exception:
            optimizer = None
    if optimizer is None:
        optimizer = torch.optim.AdamW(optim_params, lr=lr)
    grad_accum = int(cfg.get("grad_accum_steps", 4))
    max_steps = int(cfg.get("max_steps", 50))
    if max_steps <= 0:
        max_steps = 1

    model.train()
    model_device = next(model.parameters()).device
    start = time.time()
    total_loss = 0.0
    steps = 0
    tokens_seen = 0
    optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    use_autocast = device.startswith("cuda")
    grad_clip = float(cfg.get("grad_clip", 1.0))
    for batch in loader:
        batch = {k: v.to(model_device) for k, v in batch.items() if v is not None}
        with torch.autocast(device_type="cuda", dtype=torch_dtype, enabled=use_autocast):
            outputs = model(**batch)
            loss = outputs.loss / max(1, grad_accum)
        loss.backward()
        tokens_seen += int(batch["input_ids"].numel())
        steps += 1
        total_loss += float(loss.item())
        if steps % grad_accum == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if steps >= max_steps:
            break
    if steps % grad_accum != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, steps)
    elapsed = max(1e-6, time.time() - start)
    tokens_per_sec = tokens_seen / elapsed

    model.save_pretrained(adapter_dir)
    vram_peak = None
    if device.startswith("cuda"):
        try:
            vram_peak = float(torch.cuda.max_memory_allocated() / (1024**2))
        except Exception:
            vram_peak = None
    meta = {
        "run_id": run_id,
        "loss": avg_loss,
        "steps": steps,
        "samples": len(samples),
        "tokens_per_sec": tokens_per_sec,
        "vram_peak_mb": vram_peak,
        "ts": time.time(),
    }
    (adapter_dir.parent / "meta.json").write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")

    adapter_loss = None
    improvement = None
    eval_ok = True
    if eval_enabled and base_eval_samples:
        adapter_loss = _eval_loss(model, tokenizer, base_eval_samples, max_length=max_length)
        if base_loss is not None and adapter_loss is not None:
            improvement = base_loss - adapter_loss
            min_improve = float(eval_cfg.get("min_improvement", settings.get("continuous", {}).get("eval", {}).get("min_improvement", 0.0)))
            max_regress = float(eval_cfg.get("max_regression", settings.get("continuous", {}).get("eval", {}).get("max_regression", 0.0)))
            eval_ok = improvement >= min_improve and improvement >= -max_regress
        if hasattr(model, "generate"):
            prompt_text = format_chat_sample(base_eval_samples[0], backend="hf", tokenizer=tokenizer, default_system=default_system)
            try:
                gen_ids = model.generate(
                    input_ids=tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model_device),
                    max_new_tokens=int(eval_cfg.get("gen_max_new_tokens", 64)),
                    do_sample=False,
                )
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                repeat_ratio = _repeat_ratio(gen_text)
                max_repeat = float(eval_cfg.get("max_repeat_ratio", 0.9))
                if repeat_ratio > max_repeat:
                    eval_ok = False
            except Exception:
                pass
    meta.update({"base_loss": base_loss, "adapter_loss": adapter_loss, "improvement": improvement, "eval_ok": eval_ok})
    (adapter_dir.parent / "meta.json").write_text(json.dumps(meta, ensure_ascii=True), encoding="utf-8")

    state.update({"last_ts": time.time(), "last_run_id": run_id})
    _save_state(state_path, state)

    registry_path = reg_dir / "registry.json"
    if eval_ok:
        registry = {"current_adapter": str(adapter_dir), "last_run_id": run_id, "ts": time.time()}
        registry_path.write_text(json.dumps(registry, ensure_ascii=True), encoding="utf-8")
    else:
        candidate_path = reg_dir / "candidates.jsonl"
        with candidate_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": run_id, "adapter": str(adapter_dir), "improvement": improvement, "ts": time.time()}, ensure_ascii=True) + "\n")

    return HfTrainResult(
        ok=True,
        run_id=run_id,
        adapter_dir=adapter_dir,
        loss=avg_loss,
        steps=steps,
        samples=len(samples),
        tokens_per_sec=tokens_per_sec,
    )


def resolve_latest_adapter(base_dir: Path, settings: dict) -> Optional[Path]:
    reg_dir = _resolve_registry_dir(base_dir, settings)
    registry_path = reg_dir / "registry.json"
    if registry_path.exists():
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            adapter = payload.get("current_adapter")
            if adapter:
                return Path(adapter)
        except Exception:
            return None
    return None
