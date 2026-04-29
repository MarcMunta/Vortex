from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from .model_init import model_cache_status, resolve_cache_dir
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
    model_loader: str = "causal_lm"
    repo_kwargs: dict | None = None
    processor_kwargs: dict | None = None
    chat_template_kwargs: dict | None = None


class HFModel:
    is_hf = True

    def __init__(self, cfg: HFConfig):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
            except Exception:
                AutoModelForImageTextToText = None  # type: ignore[assignment]
                AutoProcessor = None  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers not available: {exc}")

        self.cfg = cfg
        self.model_loader = str(cfg.model_loader or "causal_lm").strip().lower()
        self.repo_kwargs = dict(cfg.repo_kwargs or {})
        self.processor = None
        self.chat_template_kwargs = dict(cfg.chat_template_kwargs or {})
        if self.model_loader in {"image_text_to_text", "processor_causal_lm"}:
            if AutoProcessor is None:
                raise RuntimeError(
                    "transformers processor-based loading is not available; "
                    "install a newer transformers build for Gemma 4/Gemma 3"
                )
            self.processor = AutoProcessor.from_pretrained(
                cfg.model_name,
                **self.repo_kwargs,
                **dict(cfg.processor_kwargs or {}),
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = self.processor
            if self.model_loader == "image_text_to_text":
                if AutoModelForImageTextToText is None:
                    raise RuntimeError(
                        "transformers image-text-to-text support is not available; "
                        "install a newer transformers build for Gemma 3"
                    )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    cfg.model_name,
                    **cfg.load_kwargs,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_name,
                    **cfg.load_kwargs,
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **self.repo_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **cfg.load_kwargs)
        if "device_map" not in cfg.load_kwargs:
            self.model.to(cfg.device)
        self.model.eval()
        self.device = self._input_device()
        self.base_model = self.model
        self.adapter_path = None
        # Multi-adapter support (PEFT). We serialize adapter switching per request.
        self.adapters: dict[str, str | None] = {}
        self.active_adapter_name: str | None = None
        self.adapter_max_loaded: int = 0
        self._adapter_lru: list[str] = []
        self.adapter_lock = threading.Lock()
        self._weighted_mix_cache: dict[tuple[tuple[str, float], ...], str] = {}
        self._weighted_mix_lru: list[tuple[tuple[str, float], ...]] = []

    def _input_device(self) -> torch.device:
        try:
            emb = getattr(self.model, "get_input_embeddings", None)
            if callable(emb):
                weight = emb().weight
                if getattr(weight, "device", None) is not None and weight.device.type != "meta":
                    return weight.device
        except Exception:
            pass
        try:
            device = next(self.model.parameters()).device
            if device.type != "meta":
                return device
        except Exception:
            return torch.device(str(self.cfg.device))

    def _prepare_prompt(self, prompt: str | None, messages: list[dict] | None, system: str | None) -> str:
        if messages is not None:
            return build_chat_prompt(messages, backend="hf", tokenizer=self.tokenizer, default_system=system)
        return prompt or ""

    def _normalize_text_messages(
        self,
        prompt: str | None,
        messages: list[dict] | None,
        system: str | None,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        if messages:
            has_system = any(str((msg or {}).get("role") or "").strip().lower() == "system" for msg in messages)
            if not has_system and system:
                normalized.append({"role": "system", "content": str(system).strip()})
            for raw in messages:
                if not isinstance(raw, dict):
                    continue
                role = str(raw.get("role") or "user").strip().lower() or "user"
                content = raw.get("content")
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if text is not None:
                                parts.append(str(text))
                        elif item is not None:
                            parts.append(str(item))
                    content_text = "".join(parts).strip()
                else:
                    content_text = str(content or "").strip()
                if not content_text:
                    continue
                normalized.append({"role": role, "content": content_text})
        else:
            if system:
                normalized.append({"role": "system", "content": str(system).strip()})
            normalized.append({"role": "user", "content": str(prompt or "").strip()})
        return normalized

    def _prepare_processor_messages(
        self,
        prompt: str | None,
        messages: list[dict] | None,
        system: str | None,
    ) -> list[dict]:
        prepared: list[dict] = []
        for item in self._normalize_text_messages(prompt, messages, system):
            text = str(item.get("content") or "").strip()
            if not text:
                continue
            prepared.append(
                {
                    "role": str(item.get("role") or "user"),
                    "content": [{"type": "text", "text": text}],
                }
            )
        return prepared

    def _apply_processor_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        return_dict: bool = False,
        return_tensors: str | None = None,
        add_generation_prompt: bool = True,
    ):
        if self.processor is None:
            raise RuntimeError("processor is not available")
        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if tokenize:
            kwargs["return_dict"] = return_dict
            kwargs["return_tensors"] = return_tensors
        kwargs.update(dict(self.chat_template_kwargs or {}))
        try:
            return self.processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            fallback_kwargs = {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
            if tokenize:
                fallback_kwargs["return_dict"] = return_dict
                fallback_kwargs["return_tensors"] = return_tensors
            return self.processor.apply_chat_template(messages, **fallback_kwargs)

    def _apply_tokenizer_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_tensors: str | None = None,
        add_generation_prompt: bool = True,
    ):
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("tokenizer chat template is not available")
        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if tokenize and return_tensors:
            kwargs["return_tensors"] = return_tensors
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _render_processor_prompt(
        self,
        prompt: str | None,
        messages: list[dict] | None,
        system: str | None,
    ) -> str:
        if self.processor is None:
            return ""
        proc_messages = self._prepare_processor_messages(prompt, messages, system)
        rendered = self._apply_processor_chat_template(
            proc_messages,
            tokenize=False,
        )
        return str(rendered or "")

    def _move_model_inputs(self, inputs) -> tuple[dict[str, torch.Tensor], int]:
        device = self._input_device()
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        moved_inputs: dict[str, torch.Tensor] = {}
        for key, value in dict(inputs).items():
            moved_inputs[key] = value.to(device) if hasattr(value, "to") else value
        input_ids = moved_inputs.get("input_ids")
        input_len = int(input_ids.shape[1]) if input_ids is not None else 0
        return moved_inputs, input_len

    def _postprocess_processor_response(self, response: str) -> str:
        if self.processor is not None and hasattr(self.processor, "parse_response"):
            try:
                parsed = self.processor.parse_response(response)
            except Exception:
                parsed = None
            extracted = self._extract_processor_response_text(parsed)
            if extracted:
                return extracted.strip()
        return str(response or "").strip()

    def _extract_processor_response_text(self, parsed: Any) -> str | None:
        if parsed is None:
            return None
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict):
            for key in (
                "response",
                "text",
                "output_text",
                "final",
                "final_response",
                "answer",
                "content",
            ):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            for value in parsed.values():
                candidate = self._extract_processor_response_text(value)
                if candidate:
                    return candidate
        if isinstance(parsed, (list, tuple)):
            for item in parsed:
                candidate = self._extract_processor_response_text(item)
                if candidate:
                    return candidate
        return None

    def _decode_processor_tokens(self, token_ids) -> str:
        if self.model_loader == "processor_causal_lm" and self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if self.processor is not None and hasattr(self.processor, "decode"):
            response = self.processor.decode(token_ids, skip_special_tokens=False)
            return self._postprocess_processor_response(str(response or ""))
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _encode(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        if self.processor is not None:
            proc_messages = self._prepare_processor_messages(prompt, messages, system)
            if self.model_loader == "processor_causal_lm":
                text_messages = self._normalize_text_messages(prompt, messages, system)
                tokenized = self._apply_tokenizer_chat_template(
                    text_messages,
                    tokenize=True,
                    return_tensors="pt",
                )
                if isinstance(tokenized, torch.Tensor):
                    inputs = {"input_ids": tokenized}
                else:
                    inputs = tokenized
            else:
                inputs = self._apply_processor_chat_template(
                    proc_messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            return self._move_model_inputs(inputs)

        prompt_text = self._prepare_prompt(prompt, messages, system)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        device = self._input_device()
        input_ids = input_ids.to(device)
        moved: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            moved["attention_mask"] = attention_mask
        return moved, int(input_ids.shape[1])

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
        if self.processor is not None:
            if self.model_loader == "processor_causal_lm" and self.tokenizer is not None:
                input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
                ids = input_ids.tolist()[0]
                return ids, len(ids)
            input_ids = self.processor(text=prompt, return_tensors="pt")["input_ids"]
            ids = input_ids.tolist()[0]
            return ids, len(ids)
        inputs, _ = self._encode(prompt=prompt)
        ids = inputs["input_ids"].tolist()[0]
        return ids, len(ids)

    def decode_ids(self, ids: list[int], total_len: int | None = None) -> str:
        if self.processor is not None:
            return self._decode_processor_tokens(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        max_new_tokens: int = 64,
        preserve_max_new_tokens: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        **_kwargs,
    ) -> str:
        start = time.time()
        model_inputs, prompt_len = self._encode(prompt=prompt, messages=messages, system=system)
        do_sample = temperature > 0
        max_new = max(1, int(max_new_tokens)) if preserve_max_new_tokens else self._adjust_max_new_tokens(max_new_tokens)
        for attempt in range(2):
            try:
                kwargs = dict(model_inputs)
                kwargs.update(
                    {
                        "max_new_tokens": max_new,
                        "do_sample": do_sample,
                        "temperature": temperature if do_sample else None,
                        "top_p": top_p if do_sample else None,
                        "repetition_penalty": repetition_penalty if repetition_penalty > 1.0 else None,
                        "no_repeat_ngram_size": no_repeat_ngram if no_repeat_ngram > 0 else None,
                    }
                )
                output = self.model.generate(**kwargs)
                break
            except RuntimeError as exc:
                if is_oom_error(exc) and attempt == 0:
                    clear_cuda_cache()
                    max_new = max(1, max_new // 2)
                    continue
                raise
        if self.processor is not None:
            gen_ids = output[0][prompt_len:]
            text = self._decode_processor_tokens(gen_ids)
        else:
            gen_ids = output[0][prompt_len:]
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
        preserve_max_new_tokens: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
    ) -> Iterable[str]:
        try:
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers streamer not available: {exc}")

        model_inputs, _prompt_len = self._encode(prompt=prompt, messages=messages, system=system)
        do_sample = temperature > 0
        max_new = max(1, int(max_new_tokens)) if preserve_max_new_tokens else self._adjust_max_new_tokens(max_new_tokens)
        start = time.time()
        chunks: list[str] = []
        streamer_tokenizer = self.tokenizer or self.processor
        if streamer_tokenizer is None:
            text = self.generate(
                prompt=prompt,
                messages=messages,
                system=system,
                max_new_tokens=max_new_tokens,
                preserve_max_new_tokens=preserve_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )
            if text:
                yield text
            return
        for attempt in range(2):
            try:
                streamer = TextIteratorStreamer(
                    streamer_tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
            except TypeError:
                streamer = TextIteratorStreamer(
                    streamer_tokenizer,
                    skip_special_tokens=True,
                )
            error: list[Exception] = []

            def _run():
                try:
                    kwargs = dict(model_inputs)
                    kwargs.update(
                        {
                            "max_new_tokens": max_new,
                            "do_sample": do_sample,
                            "temperature": temperature if do_sample else None,
                            "top_p": top_p if do_sample else None,
                            "repetition_penalty": repetition_penalty if repetition_penalty > 1.0 else None,
                            "no_repeat_ngram_size": no_repeat_ngram if no_repeat_ngram > 0 else None,
                            "streamer": streamer,
                        }
                    )
                    self.model.generate(**kwargs)
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
                model = PeftModel.from_pretrained(
                    base,
                    adapter_path,
                    adapter_name=adapter_name,
                    autocast_adapter_dtype=False,
                )
            except TypeError:
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
                    peft_model.load_adapter(
                        str(path),
                        adapter_name=name,
                        autocast_adapter_dtype=False,
                    )
                except TypeError:
                    try:
                        peft_model.load_adapter(str(path), adapter_name=name)
                    except TypeError:
                        peft_model.load_adapter(str(path))
                self.model = peft_model
            else:
                base = getattr(self, "base_model", None) or self.model
                try:
                    self.model = PeftModel.from_pretrained(
                        base,
                        str(path),
                        adapter_name=name,
                        autocast_adapter_dtype=False,
                    )
                except TypeError:
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

    def _weighted_mix_cache_limit(self) -> int:
        try:
            limit = int(getattr(self, "adapter_max_loaded", 0) or 0)
        except Exception:
            limit = 0
        if limit <= 0:
            return 8
        return max(2, min(16, limit))

    def _touch_weighted_mix_lru(self, key: tuple[tuple[str, float], ...]) -> None:
        try:
            self._weighted_mix_lru.remove(key)
        except ValueError:
            pass
        self._weighted_mix_lru.append(key)

    def _evict_weighted_mix_cache(self) -> None:
        limit = self._weighted_mix_cache_limit()
        while len(self._weighted_mix_lru) > limit:
            victim_key = self._weighted_mix_lru.pop(0)
            victim_name = self._weighted_mix_cache.pop(victim_key, None)
            if not victim_name:
                continue
            if victim_name == getattr(self, "active_adapter_name", None):
                continue
            if hasattr(self.model, "delete_adapter"):
                try:
                    self.model.delete_adapter(victim_name)
                except Exception:
                    pass
            self.adapters.pop(victim_name, None)
            try:
                self._adapter_lru.remove(victim_name)
            except ValueError:
                pass

    def set_weighted_adapters(self, adapter_weights: dict[str, float]) -> bool:
        """Best-effort adapter mixing using PEFT. Falls back to top-1 when unsupported."""
        if not adapter_weights:
            return False
        cleaned: dict[str, float] = {}
        for raw_name, raw_w in adapter_weights.items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            try:
                w = float(raw_w)
            except Exception:
                continue
            if w <= 0:
                continue
            cleaned[name] = cleaned.get(name, 0.0) + w
        if not cleaned:
            return False
        if len(cleaned) == 1:
            return self.set_adapter(next(iter(cleaned.keys())))
        total = sum(cleaned.values())
        if total <= 0:
            return False
        normalized = {k: float(v) / float(total) for k, v in cleaned.items()}
        top1 = max(normalized.items(), key=lambda kv: kv[1])[0]

        if not hasattr(self.model, "add_weighted_adapter") or not hasattr(self.model, "set_adapter"):
            return self.set_adapter(top1)

        key = tuple(sorted((name, round(float(weight), 4)) for name, weight in normalized.items()))
        cached = self._weighted_mix_cache.get(key)
        peft_cfg = getattr(self.model, "peft_config", None)
        if cached and isinstance(peft_cfg, dict) and cached in peft_cfg:
            try:
                self.model.set_adapter(cached)
                self.active_adapter_name = cached
                self.adapter_path = None
                self.adapters.setdefault(cached, None)
                self._touch_adapter_lru(cached)
                self._touch_weighted_mix_lru(key)
                return True
            except Exception:
                pass

        digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:10]
        virtual_name = cached or f"mix_{digest}"

        adapters = [name for name, _w in key]
        weights = [float(w) for _name, w in key]
        try:
            add_fn = getattr(self.model, "add_weighted_adapter")
            try:
                add_fn(adapters=adapters, weights=weights, adapter_name=virtual_name, combination_type="linear")
            except TypeError:
                try:
                    add_fn(adapters, weights, adapter_name=virtual_name, combination_type="linear")
                except TypeError:
                    try:
                        add_fn(adapters, weights, virtual_name)
                    except TypeError:
                        add_fn(adapters, weights)
            self.model.set_adapter(virtual_name)
        except Exception:
            return self.set_adapter(top1)

        self._weighted_mix_cache[key] = virtual_name
        self._touch_weighted_mix_lru(key)
        self._evict_weighted_mix_cache()

        self.adapters.setdefault(virtual_name, None)
        self.active_adapter_name = virtual_name
        self.adapter_path = None
        self._touch_adapter_lru(virtual_name)
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
    device_map: str | dict | None,
    offload_folder: str | None,
    use_safetensors: bool | None,
) -> dict:
    load_kwargs: dict = {"torch_dtype": torch_dtype}
    if use_safetensors is not None:
        load_kwargs["use_safetensors"] = bool(use_safetensors)
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    if max_memory:
        load_kwargs["max_memory"] = max_memory
    if offload_folder:
        load_kwargs["offload_folder"] = str(offload_folder)
    if load_in_4bit or load_in_8bit:
        load_kwargs["load_in_4bit"] = load_in_4bit
        load_kwargs["load_in_8bit"] = load_in_8bit
    if device_map:
        load_kwargs["device_map"] = device_map
    elif (load_in_4bit or load_in_8bit) and device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
    elif max_memory and device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
    if "device_map" in load_kwargs:
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        load_kwargs.setdefault("offload_state_dict", True)
    return load_kwargs


def _try_load(cfg: HFConfig) -> HFModel:
    return HFModel(cfg)


def _resolve_hf_cache_settings(core: dict) -> tuple[Path, bool]:
    cache_dir = resolve_cache_dir(core.get("hf_cache_dir") or str(Path(".") / "data" / "models" / "hf-cache"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    offline_forced = str(os.getenv("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    offline_forced = offline_forced or str(os.getenv("TRANSFORMERS_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    offline_forced = offline_forced or bool(core.get("hf_local_files_only", False))
    return cache_dir, offline_forced


def _hf_repo_cache_dirs(cache_dir: Path, model_name: str) -> list[Path]:
    repo_name = f"models--{str(model_name).replace('/', '--')}"
    dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in (cache_dir / repo_name, cache_dir / "hub" / repo_name):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(candidate)
    return dirs


def _best_cached_hf_snapshot(cache_dir: Path, model_name: str) -> Path | None:
    best_snapshot: Path | None = None
    best_score: tuple[int, int, str] | None = None
    for repo_dir in _hf_repo_cache_dirs(cache_dir, model_name):
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        for snapshot_dir in snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            try:
                names = {path.name for path in snapshot_dir.iterdir() if not path.name.endswith(".materializing")}
            except Exception:
                continue
            score = 0
            for required_name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
                if required_name in names:
                    score += 5
            if "model.safetensors.index.json" in names:
                score += 20
            if "model.safetensors" in names or "pytorch_model.bin" in names:
                score += 40
            shard_numbers: set[int] = set()
            score += sum(
                10
                for name in names
                if name.startswith("model-")
                and (name.endswith(".safetensors") or name.endswith(".bin"))
            )
            for name in names:
                if not name.startswith("model-"):
                    continue
                if "-of-" not in name:
                    continue
                shard_part = name.split("-of-", 1)[0].split("-")[-1]
                with contextlib.suppress(Exception):
                    shard_numbers.add(int(shard_part))
            current_score = (score, len(shard_numbers), len(names), str(snapshot_dir))
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_snapshot = snapshot_dir
    return best_snapshot


def _decode_windows_reparse_target(output: str) -> str | None:
    hex_pairs = []
    for line in str(output or "").splitlines():
        if ":" not in line:
            continue
        _, right = line.split(":", 1)
        hex_pairs.extend(part for part in right.strip().split() if len(part) == 2 and all(ch in "0123456789abcdefABCDEF" for ch in part))
    if not hex_pairs:
        return None
    try:
        raw = bytes.fromhex("".join(hex_pairs))
    except Exception:
        return None
    marker = b"../"
    idx = raw.find(marker)
    if idx < 0:
        marker = b"..\\"
        idx = raw.find(marker)
    if idx < 0:
        return None
    rel = raw[idx:].split(b"\x00", 1)[0]
    try:
        return rel.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _replace_with_local_blob(path: Path, target: Path) -> bool:
    temp_path = path.with_name(path.name + ".materializing")
    with contextlib.suppress(Exception):
        if temp_path.exists():
            temp_path.unlink()
    try:
        os.link(target, temp_path)
        os.replace(temp_path, path)
        return True
    except Exception:
        with contextlib.suppress(Exception):
            if temp_path.exists():
                temp_path.unlink()
    try:
        shutil.copy2(target, temp_path)
        os.replace(temp_path, path)
        return True
    except Exception:
        with contextlib.suppress(Exception):
            if temp_path.exists():
                temp_path.unlink()
        return False


def _prepare_local_hf_snapshot(cache_dir: Path, model_name: str) -> Path | None:
    snapshot_dir = _best_cached_hf_snapshot(cache_dir, model_name)
    if snapshot_dir is None:
        return None
    if sys.platform.startswith("win"):
        for temp_path in snapshot_dir.glob("*.materializing"):
            with contextlib.suppress(Exception):
                temp_path.unlink()
        for path in snapshot_dir.iterdir():
            if path.name.endswith(".materializing"):
                continue
            try:
                query = subprocess.run(
                    ["cmd.exe", "/c", "fsutil", "reparsepoint", "query", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception:
                continue
            if query.returncode != 0:
                continue
            rel_target = _decode_windows_reparse_target((query.stdout or "") + "\n" + (query.stderr or ""))
            if not rel_target:
                continue
            target = (path.parent / rel_target.replace("/", os.sep)).resolve()
            if not target.exists() or not target.is_file():
                continue
            _replace_with_local_blob(path, target)
    critical_files = (
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for name in critical_files:
        path = snapshot_dir / name
        if not path.exists():
            return None
        with contextlib.suppress(Exception):
            if path.stat().st_size <= 0:
                return None
    return snapshot_dir


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
    device_map = core.get("hf_device_map")
    offload_folder = core.get("hf_offload_folder")
    if isinstance(offload_folder, str) and offload_folder:
        try:
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
        except Exception:
            offload_folder = None
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
    model_loader = str(core.get("hf_model_loader") or "causal_lm").strip().lower()
    processor_kwargs: dict[str, Any] = {}
    chat_template_kwargs: dict[str, Any] = {}
    cache_dir, offline_forced = _resolve_hf_cache_settings(core)
    local_snapshot = _prepare_local_hf_snapshot(cache_dir, str(model_name))
    cache_status = model_cache_status(str(model_name), cache_dir)
    repo_cached = bool(local_snapshot) or bool(cache_status.get("cached"))
    if offline_forced and not repo_cached:
        raise RuntimeError(
            "hf model cache missing for offline load: "
            f"{model_name} in {cache_dir}. "
            "Run explicitly: python -m c3rnt2.model_init "
            f"--model {model_name} --cache-dir {cache_dir} --download"
        )
    model_source = str(local_snapshot) if local_snapshot is not None else str(model_name)
    use_cache_lookup = local_snapshot is None
    base_repo_kwargs: dict[str, Any] = {}
    if use_cache_lookup:
        base_repo_kwargs["cache_dir"] = str(cache_dir)
        if offline_forced:
            base_repo_kwargs["local_files_only"] = True
    processor_padding_side = core.get("hf_processor_padding_side")
    if processor_padding_side:
        processor_kwargs["padding_side"] = str(processor_padding_side)
    if core.get("hf_enable_thinking") is not None:
        chat_template_kwargs["enable_thinking"] = bool(core.get("hf_enable_thinking"))

    def _build_attempt(
        *,
        use_quant_4bit: bool,
        use_quant_8bit: bool,
        attn_value: str | None,
        max_memory_value: Any,
        device_map_value: Any,
        offload_folder_value: Any,
        use_local_files_only: bool,
        force_device: str | None = None,
        force_dtype: torch.dtype | None = None,
    ) -> HFConfig:
        repo_kwargs = dict(base_repo_kwargs)
        if use_local_files_only:
            repo_kwargs["local_files_only"] = True
        load_kwargs = _build_load_kwargs(
            force_dtype or torch_dtype,
            force_device or device,
            use_quant_4bit,
            use_quant_8bit,
            attn_value,
            max_memory_value,
            device_map_value,
            offload_folder_value,
            use_safetensors,
        )
        if use_cache_lookup:
            load_kwargs["cache_dir"] = str(cache_dir)
        if use_local_files_only and use_cache_lookup:
            load_kwargs["local_files_only"] = True
        return HFConfig(
            model_name=model_source,
            device=force_device or device,
            dtype=force_dtype or torch_dtype,
            load_kwargs=load_kwargs,
            model_loader=model_loader,
            repo_kwargs=repo_kwargs,
            processor_kwargs=dict(processor_kwargs),
            chat_template_kwargs=dict(chat_template_kwargs),
        )

    attempt_specs: list[HFConfig] = []
    local_modes = [False] if local_snapshot is not None else ([True] if offline_forced else ([True, False] if repo_cached else [False]))

    for use_local_files_only in local_modes:
        if device_map or (max_memory and device.startswith("cuda")):
            attempt_specs.append(
                _build_attempt(
                    use_quant_4bit=False,
                    use_quant_8bit=False,
                    attn_value=attn_impl,
                    max_memory_value=max_memory,
                    device_map_value=device_map,
                    offload_folder_value=offload_folder,
                    use_local_files_only=use_local_files_only,
                )
            )
        if quant_requested and quant_available:
            attempt_specs.append(
                _build_attempt(
                    use_quant_4bit=load_in_4bit,
                    use_quant_8bit=load_in_8bit,
                    attn_value=attn_impl,
                    max_memory_value=max_memory,
                    device_map_value=device_map,
                    offload_folder_value=offload_folder,
                    use_local_files_only=use_local_files_only,
                )
            )
            if attn_impl:
                attempt_specs.append(
                    _build_attempt(
                        use_quant_4bit=load_in_4bit,
                        use_quant_8bit=load_in_8bit,
                        attn_value=None,
                        max_memory_value=max_memory,
                        device_map_value=device_map,
                        offload_folder_value=offload_folder,
                        use_local_files_only=use_local_files_only,
                    )
                )
        attempt_specs.append(
            _build_attempt(
                use_quant_4bit=False,
                use_quant_8bit=False,
                attn_value=attn_impl,
                max_memory_value=max_memory,
                device_map_value=device_map,
                offload_folder_value=offload_folder,
                use_local_files_only=use_local_files_only,
            )
        )
        if attn_impl:
            attempt_specs.append(
                _build_attempt(
                    use_quant_4bit=False,
                    use_quant_8bit=False,
                    attn_value=None,
                    max_memory_value=max_memory,
                    device_map_value=device_map,
                    offload_folder_value=offload_folder,
                    use_local_files_only=use_local_files_only,
                )
            )
        if device.startswith("cuda"):
            attempt_specs.append(
                _build_attempt(
                    use_quant_4bit=False,
                    use_quant_8bit=False,
                    attn_value=None,
                    max_memory_value=None,
                    device_map_value=None,
                    offload_folder_value=None,
                    use_local_files_only=use_local_files_only,
                    force_device="cpu",
                    force_dtype=torch.float32,
                )
            )

    attempts: list[HFConfig] = []
    seen_attempt_keys: set[str] = set()
    for cfg in attempt_specs:
        def _json_safe(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            return value

        key = json.dumps(
            {
                "device": cfg.device,
                "dtype": str(cfg.dtype),
                "loader": cfg.model_loader,
                "repo_kwargs": _json_safe(cfg.repo_kwargs),
                "processor_kwargs": _json_safe(cfg.processor_kwargs),
                "load_kwargs": _json_safe(cfg.load_kwargs),
            },
            sort_keys=True,
            default=str,
        )
        if key in seen_attempt_keys:
            continue
        seen_attempt_keys.add(key)
        attempts.append(cfg)

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
    adapter_event: dict[str, Any] = {
        "event": "hf_adapter_autoload",
        "model": str(model_name),
        "use_latest": bool(use_latest),
        "registry_dir": str((settings.get("hf_train", {}) or {}).get("registry_dir") or "data/registry/hf_train"),
        "loaded": False,
    }
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
            try:
                model.model = PeftModel.from_pretrained(base_model, adapter_path, autocast_adapter_dtype=False)
            except TypeError:
                model.model = PeftModel.from_pretrained(base_model, adapter_path)
            try:
                model.adapter_path = adapter_path
            except Exception:
                pass
            if merge_adapter and hasattr(model.model, "merge_and_unload"):
                model.model = model.model.merge_and_unload()
        adapter_event.update({"loaded": True, "adapter_path": adapter_path})
    elif use_latest:
        adapter_event.update({"warning": "latest_adapter_not_found"})
    if use_latest:
        try:
            _log_infer_stats(Path("."), adapter_event)
        except Exception:
            pass
    used_quant = False
    if hasattr(model, "cfg"):
        used_quant = bool(model.cfg.load_kwargs.get("load_in_4bit") or model.cfg.load_kwargs.get("load_in_8bit"))
    model.quant_fallback = bool(quant_requested) and not used_quant
    return model

