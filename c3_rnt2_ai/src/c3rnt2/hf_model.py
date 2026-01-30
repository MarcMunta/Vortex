from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class HFConfig:
    model_name: str
    device: str
    dtype: torch.dtype


class HFModel:
    is_hf = True

    def __init__(self, cfg: HFConfig):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers not available: {exc}")

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
        self.model.to(cfg.device)
        self.model.eval()
        self.device = torch.device(cfg.device)

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
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        **_kwargs,
    ) -> str:
        input_ids = self._encode(prompt)
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
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
    ) -> Iterable[str]:
        try:
            from transformers import TextIteratorStreamer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers streamer not available: {exc}")

        input_ids = self._encode(prompt)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = core.get("dtype")
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    cfg = HFConfig(model_name=str(model_name), device=device, dtype=torch_dtype)
    return HFModel(cfg)
