from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class TensorRTConfig:
    engine_dir: Path
    tokenizer_name: str


class TensorRTModel:
    is_tensorrt = True

    def __init__(self, cfg: TensorRTConfig):
        self.cfg = cfg
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"transformers not available: {exc}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        runtime = importlib.import_module("tensorrt_llm.runtime")
        if not hasattr(runtime, "ModelRunner"):
            raise RuntimeError("tensorrt_llm runtime does not expose ModelRunner")
        self.runner = runtime.ModelRunner.from_dir(str(cfg.engine_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"].to(self.device)

    def encode_prompt(self, prompt: str):
        ids = self._encode(prompt).tolist()[0]
        return ids, len(ids)

    def decode_ids(self, ids: list[int], total_len: int | None = None) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _extract_output_ids(self, outputs, input_len: int) -> list[int]:
        if isinstance(outputs, dict):
            if "output_ids" in outputs:
                out_ids = outputs["output_ids"]
            elif "output_tokens" in outputs:
                out_ids = outputs["output_tokens"]
            else:
                out_ids = next(iter(outputs.values()))
        else:
            out_ids = outputs
        if torch.is_tensor(out_ids):
            out_ids = out_ids.tolist()
        if isinstance(out_ids, list) and out_ids and isinstance(out_ids[0], list):
            out_ids = out_ids[0]
        if isinstance(out_ids, list):
            return out_ids[input_len:]
        return []

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
        if messages is not None:
            prompt = "\n".join(str(m.get("content", "")) for m in messages)
        prompt = prompt or ""
        input_ids = self._encode(prompt)
        input_len = input_ids.shape[1]
        outputs = self.runner.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram,
        )
        gen_ids = self._extract_output_ids(outputs, input_len)
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
        text = self.generate(
            prompt=prompt,
            messages=messages,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
        )
        if text:
            yield text


def load_tensorrt_model(settings: dict) -> TensorRTModel:
    core = settings.get("core", {}) or {}
    engine_dir = core.get("tensorrt_engine_dir") or core.get("tensorrt_engine_path")
    if not engine_dir:
        raise ValueError("core.tensorrt_engine_dir is required for tensorrt backend")
    tokenizer_name = core.get("tensorrt_tokenizer") or core.get("hf_model")
    if not tokenizer_name:
        raise ValueError("core.tensorrt_tokenizer or core.hf_model required for tensorrt backend")
    try:
        importlib.import_module("tensorrt_llm")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorrt_llm not installed. Install TensorRT-LLM or set core.backend=hf/vortex."
        ) from exc
    cfg = TensorRTConfig(engine_dir=Path(engine_dir), tokenizer_name=str(tokenizer_name))
    return TensorRTModel(cfg)

