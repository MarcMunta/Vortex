from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .prompting.chat_format import build_chat_prompt
from .utils.oom import clear_cuda_cache, is_oom_error
from .utils.vram import get_vram_free_mb, recommended_max_new_tokens, should_reduce_decode


def _log_infer_stats(base_dir: Path, payload: dict) -> None:
    log_dir = base_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "hf_infer.jsonl"
    meta_path = log_dir / "hf_infer_meta.json"
    payload = dict(payload)
    payload.setdefault("ts", time.time())
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    try:
        meta_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass

@dataclass
class HFConfig:
    model_name: str
    device: str
    dtype: torch.dtype
    load_kwargs: dict


class HFModel:
    is_hf = True

    def __init__(self, cfg: HFConfig):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers not available: {exc}")

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **cfg.load_kwargs)
        if "device_map" not in cfg.load_kwargs:
            self.model.to(cfg.device)
        self.model.eval()
        self.device = torch.device(cfg.device)
        self.base_model = self.model
        self.adapter_path = None
        # Multi-adapter support (PEFT). We serialize adapter switching per request.
        self.adapters: dict[str, str] = {}
        self.active_adapter_name: str | None = None
        self.adapter_max_loaded: int = 0
        self._adapter_lru: list[str] = []
        self.adapter_lock = threading.Lock()

    def _prepare_prompt(self, prompt: str | None, messages: list[dict] | None, system: str | None) -> str:
        if messages is not None:
            return build_chat_prompt(messages, backend="hf", tokenizer=self.tokenizer, default_system=system)
        return prompt or ""

    def _encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"].to(self.cfg.device)

    def _adjust_max_new_tokens(self, max_new_tokens: int) -> int:
        cfg = getattr(self, "vram_cfg", {}) or {}
        threshold_mb = float(cfg.get("threshold_mb", 0.0))
        floor = int(cfg.get("floor_tokens", 16))
        ceil = int(cfg.get("ceil_tokens", max_new_tokens))
        max_new = int(max_new_tokens)
        if threshold_mb <= 0:
            return max(1, max_new)
        free_mb = get_vram_free_mb()
        if should_reduce_decode(free_mb, threshold_mb):
            return recommended_max_new_tokens(max_new, free_mb, floor, ceil)
        return max(1, max_new)

    def encode_prompt(self, prompt: str):
        ids = self._encode(prompt).tolist()[0]
        return ids, len(ids)

    def decode_ids(self, ids: list[int], total_len: int | None = None) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        **_kwargs,
    ) -> str:
        start = time.time()
        prompt_text = self._prepare_prompt(prompt, messages, system)
        input_ids = self._encode(prompt_text)
        do_sample = temperature > 0
        max_new = self._adjust_max_new_tokens(max_new_tokens)
        for attempt in range(2):
            try:
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    repetition_penalty=repetition_penalty if repetition_penalty > 1.0 else None,
                    no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram > 0 else None,
                )
                break
            except RuntimeError as exc:
                if is_oom_error(exc) and attempt == 0:
                    clear_cuda_cache()
                    max_new = max(1, max_new // 2)
                    continue
                raise
        gen_ids = output[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        elapsed = max(1e-6, time.time() - start)
        vram_peak = None
        if torch.cuda.is_available():
            try:
                vram_peak = float(torch.cuda.max_memory_allocated() / (1024**2))
            except Exception:
                vram_peak = None
        base_dir = getattr(self, "base_dir", Path("."))
        _log_infer_stats(
            Path(base_dir),
            {
                "tokens": int(gen_ids.numel()),
                "tokens_per_sec": float(gen_ids.numel()) / elapsed,
                "vram_peak_mb": vram_peak,
                "adapter": getattr(self, "active_adapter_name", None),
            },
        )
        return text

    def stream_generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
    ) -> Iterable[str]:
        prompt_text = self._prepare_prompt(prompt, messages, system)
        try:
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers streamer not available: {exc}")

        input_ids = self._encode(prompt_text)
        do_sample = temperature > 0
        max_new = self._adjust_max_new_tokens(max_new_tokens)
        start = time.time()
        chunks: list[str] = []
        for attempt in range(2):
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            error: list[Exception] = []

            def _run():
                try:
                    self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        top_p=top_p if do_sample else None,
                        repetition_penalty=repetition_penalty if repetition_penalty > 1.0 else None,
                        no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram > 0 else None,
                        streamer=streamer,
                    )
                except Exception as exc:  # pragma: no cover - captured in error list
                    error.append(exc)

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            for chunk in streamer:
                if chunk:
                    chunks.append(chunk)
                    yield chunk
            thread.join(timeout=0.1)
            if error:
                exc = error[0]
                if is_oom_error(exc) and attempt == 0 and not chunks:
                    clear_cuda_cache()
                    max_new = max(1, max_new // 2)
                    continue
                raise exc
            break
        elapsed = max(1e-6, time.time() - start)
        try:
            count = len(self.tokenizer("".join(chunks), add_special_tokens=False)["input_ids"])
        except Exception:
            count = len("".join(chunks).split())
        vram_peak = None
        if torch.cuda.is_available():
            try:
                vram_peak = float(torch.cuda.max_memory_allocated() / (1024**2))
            except Exception:
                vram_peak = None
        base_dir = getattr(self, "base_dir", Path("."))
        _log_infer_stats(
            Path(base_dir),
            {
                "tokens": int(count),
                "tokens_per_sec": float(count) / elapsed,
                "vram_peak_mb": vram_peak,
                "stream": True,
                "adapter": getattr(self, "active_adapter_name", None),
            },
        )

    def load_adapter(self, adapter_path: str, merge: bool = False) -> bool:
        if not adapter_path:
            return False
        if getattr(self, "adapter_path", None) == adapter_path:
            return False
        with self.adapter_lock:
            try:
                from peft import PeftModel  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"peft not available for adapter load: {exc}")
            base = getattr(self, "base_model", None) or self.model
            # Keep backward compatibility: treat "latest" as a special adapter name.
            adapter_name = "latest"
            try:
                model = PeftModel.from_pretrained(base, adapter_path, adapter_name=adapter_name)
            except TypeError:
                model = PeftModel.from_pretrained(base, adapter_path)
            if merge and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
                self.base_model = model
            self.model = model
            self.adapter_path = adapter_path
            self.adapters[str(adapter_name)] = str(adapter_path)
            self.active_adapter_name = str(adapter_name)
            self._touch_adapter_lru(str(adapter_name))
            return True

    def add_adapter(self, name: str, path: str) -> bool:
        name = str(name or "").strip()
        if not name or not path:
            return False
        with self.adapter_lock:
            existing = self.adapters.get(name)
            if existing and existing == str(path):
                self._touch_adapter_lru(name)
                return False
            try:
                from peft import PeftModel  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"peft not available for adapter load: {exc}")
            if getattr(self.model, "peft_config", None) is not None and hasattr(self.model, "load_adapter"):
                peft_model = self.model
                try:
                    peft_model.load_adapter(str(path), adapter_name=name)
                except TypeError:
                    peft_model.load_adapter(str(path))
                self.model = peft_model
            else:
                base = getattr(self, "base_model", None) or self.model
                try:
                    self.model = PeftModel.from_pretrained(base, str(path), adapter_name=name)
                except TypeError:
                    # Older/unknown PEFT: adapter name not supported.
                    self.model = PeftModel.from_pretrained(base, str(path))
            self.adapters[name] = str(path)
            self._touch_adapter_lru(name)
            self._enforce_adapter_limit()
            return True

    def set_adapter(self, name: str) -> bool:
        name = str(name or "").strip()
        if not name:
            return False
        with self.adapter_lock:
            if getattr(self, "active_adapter_name", None) == name:
                self._touch_adapter_lru(name)
                return False
            if hasattr(self.model, "set_adapter"):
                self.model.set_adapter(name)
            else:
                return False
            self.active_adapter_name = name
            if name in self.adapters:
                self.adapter_path = self.adapters.get(name)
            self._touch_adapter_lru(name)
            return True

    def _touch_adapter_lru(self, name: str) -> None:
        try:
            self._adapter_lru.remove(name)
        except ValueError:
            pass
        self._adapter_lru.append(name)

    def _enforce_adapter_limit(self) -> None:
        limit = int(getattr(self, "adapter_max_loaded", 0) or 0)
        if limit <= 0:
            return
        # Best-effort eviction: if PEFT can't delete adapters, we keep them loaded.
        while len(self.adapters) > limit:
            victim = None
            for cand in list(self._adapter_lru):
                if cand != getattr(self, "active_adapter_name", None):
                    victim = cand
                    break
            if victim is None:
                return
            removed = False
            if hasattr(self.model, "delete_adapter"):
                try:
                    self.model.delete_adapter(victim)
                    removed = True
                except Exception:
                    removed = False
            if removed:
                self.adapters.pop(victim, None)
                try:
                    self._adapter_lru.remove(victim)
                except ValueError:
                    pass
                continue
            # No reliable eviction available.
            return


def _build_load_kwargs(
    torch_dtype: torch.dtype,
    device: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    attn_impl: str | None,
    max_memory: dict | str | None,
    use_safetensors: bool | None,
) -> dict:
    load_kwargs: dict = {"torch_dtype": torch_dtype}
    if use_safetensors is not None:
        load_kwargs["use_safetensors"] = bool(use_safetensors)
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    if load_in_4bit or load_in_8bit:
        load_kwargs["load_in_4bit"] = load_in_4bit
        load_kwargs["load_in_8bit"] = load_in_8bit
        load_kwargs["device_map"] = "auto" if device.startswith("cuda") else "cpu"
    return load_kwargs


def _try_load(cfg: HFConfig) -> HFModel:
    return HFModel(cfg)


def load_hf_model(settings: dict) -> HFModel:
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    core = settings.get("core", {}) or {}
    model_name = core.get("hf_model")
    if not model_name:
        raise ValueError("core.hf_model is required for hf backend")
    device = str(core.get("hf_device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = core.get("dtype")
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    attn_impl = core.get("hf_attn_implementation")
    if not attn_impl and core.get("hf_attn_auto"):
        try:
            import flash_attn  # type: ignore  # noqa: F401

            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"
    load_in_4bit = bool(core.get("hf_load_in_4bit"))
    load_in_8bit = bool(core.get("hf_load_in_8bit"))
    max_memory = core.get("hf_max_memory")
    use_safetensors = core.get("hf_use_safetensors")
    use_safetensors = bool(use_safetensors) if use_safetensors is not None else None
    quant_requested = bool(load_in_4bit or load_in_8bit)
    quant_available = False
    if quant_requested:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
            quant_available = True
        except Exception:
            quant_available = False

    attempts: list[HFConfig] = []
    if quant_requested and quant_available:
        attempts.append(
            HFConfig(
                model_name=str(model_name),
                device=device,
                dtype=torch_dtype,
                load_kwargs=_build_load_kwargs(torch_dtype, device, load_in_4bit, load_in_8bit, attn_impl, max_memory, use_safetensors),
            )
        )
        if attn_impl:
            attempts.append(
                HFConfig(
                    model_name=str(model_name),
                    device=device,
                    dtype=torch_dtype,
                    load_kwargs=_build_load_kwargs(torch_dtype, device, load_in_4bit, load_in_8bit, None, max_memory, use_safetensors),
                )
            )
    # Non-quant fallback
    attempts.append(
        HFConfig(
            model_name=str(model_name),
            device=device,
            dtype=torch_dtype,
            load_kwargs=_build_load_kwargs(torch_dtype, device, False, False, attn_impl, max_memory, use_safetensors),
        )
    )
    if attn_impl:
        attempts.append(
            HFConfig(
                model_name=str(model_name),
                device=device,
                dtype=torch_dtype,
                load_kwargs=_build_load_kwargs(torch_dtype, device, False, False, None, max_memory, use_safetensors),
            )
        )
    # CPU fallback
    if device.startswith("cuda"):
        attempts.append(
            HFConfig(
                model_name=str(model_name),
                device="cpu",
                dtype=torch.float32,
                load_kwargs=_build_load_kwargs(torch.float32, "cpu", False, False, None, None, use_safetensors),
            )
        )

    last_exc: Exception | None = None
    model: HFModel | None = None
    for cfg in attempts:
        try:
            model = _try_load(cfg)
            break
        except Exception as exc:
            last_exc = exc
            continue
    if model is None:
        raise RuntimeError(f"hf model load failed: {last_exc}")
    model.vram_cfg = {
        "threshold_mb": float(core.get("vram_threshold_mb", 0.0)),
        "floor_tokens": int(core.get("vram_floor_tokens", 16)),
        "ceil_tokens": int(core.get("vram_ceil_tokens", 512)),
    }
    adapter_path = core.get("hf_adapter_path")
    use_latest = bool(core.get("hf_use_latest_adapter", False))
    merge_adapter = bool(core.get("hf_merge_adapter", False))
    if adapter_path is None and use_latest:
        try:
            from .training.hf_qlora import resolve_latest_adapter

            adapter_path = resolve_latest_adapter(Path("."), settings)
        except Exception:
            adapter_path = None
    if adapter_path:
        adapter_path = str(adapter_path)
        if hasattr(model, "load_adapter"):
            model.load_adapter(adapter_path, merge=merge_adapter)
        else:
            try:
                from peft import PeftModel  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"peft not available for adapter load: {exc}")
            base_model = getattr(model, "model", model)
            model.model = PeftModel.from_pretrained(base_model, adapter_path)
            try:
                model.adapter_path = adapter_path
            except Exception:
                pass
            if merge_adapter and hasattr(model.model, "merge_and_unload"):
                model.model = model.model.merge_and_unload()
    used_quant = False
    if hasattr(model, "cfg"):
        used_quant = bool(model.cfg.load_kwargs.get("load_in_4bit") or model.cfg.load_kwargs.get("load_in_8bit"))
    model.quant_fallback = bool(quant_requested) and not used_quant
    return model

