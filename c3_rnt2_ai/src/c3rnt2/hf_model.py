from __future__ import annotations

import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .prompting.chat_format import build_chat_prompt

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

    def _prepare_prompt(self, prompt: str | None, messages: list[dict] | None, system: str | None) -> str:
        if messages is not None:
            return build_chat_prompt(messages, backend="hf", tokenizer=self.tokenizer, default_system=system)
        return prompt or ""

    def _encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"].to(self.cfg.device)


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
        prompt_text = self._prepare_prompt(prompt, messages, system)
        input_ids = self._encode(prompt_text)
        do_sample = temperature > 0
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty if repetition_penalty > 1.0 else None,
            no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram > 0 else None,
        )
        gen_ids = output[0][input_ids.shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

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
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        def _run():
            self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty if repetition_penalty > 1.0 else None,
                no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram > 0 else None,
                streamer=streamer,
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join(timeout=0.1)


def load_hf_model(settings: dict) -> HFModel:
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
    load_kwargs: dict = {"torch_dtype": torch_dtype}
    attn_impl = core.get("hf_attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl
    load_in_4bit = bool(core.get("hf_load_in_4bit"))
    load_in_8bit = bool(core.get("hf_load_in_8bit"))
    if load_in_4bit or load_in_8bit:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
            load_kwargs["load_in_4bit"] = load_in_4bit
            load_kwargs["load_in_8bit"] = load_in_8bit
            load_kwargs["device_map"] = "auto" if device == "cuda" else "cpu"
        except Exception:
            load_in_4bit = False
            load_in_8bit = False
    cfg = HFConfig(model_name=str(model_name), device=device, dtype=torch_dtype, load_kwargs=load_kwargs)
    model = HFModel(cfg)
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
        try:
            from peft import PeftModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"peft not available for adapter load: {exc}")
        adapter_path = str(adapter_path)
        model.model = PeftModel.from_pretrained(model.model, adapter_path)
        if merge_adapter and hasattr(model.model, "merge_and_unload"):
            model.model = model.model.merge_and_unload()
    if load_in_4bit or load_in_8bit:
        model.quant_fallback = False
    else:
        if bool(core.get("hf_load_in_4bit")) or bool(core.get("hf_load_in_8bit")):
            model.quant_fallback = True
    return model

