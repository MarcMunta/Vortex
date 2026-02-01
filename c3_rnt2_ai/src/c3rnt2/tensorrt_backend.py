from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class TRTConfig:
    engine_path: str
    tokenizer_path: str | None
    device: str


class TRTModel:
    is_tensorrt = True

    def __init__(self, cfg: TRTConfig):
        try:
            import tensorrt_llm  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"tensorrt backend not available: {exc}")
        if not cfg.engine_path:
            raise RuntimeError("core.trt_engine_path required for tensorrt backend")
        self.cfg = cfg
        self.device = torch.device(cfg.device) if torch is not None else "cpu"
        # Placeholder: implement real TRT-LLM session wiring here.
        raise RuntimeError("tensorrt backend detected but not configured; set core.trt_engine_path and install tensorrt_llm")

    def encode_prompt(self, prompt: str):  # pragma: no cover
        raise RuntimeError("tensorrt backend not configured")

    def decode_ids(self, ids: list[int], total_len: int | None = None) -> str:  # pragma: no cover
        raise RuntimeError("tensorrt backend not configured")

    def generate(self, prompt: str, max_new_tokens: int = 64, **_kwargs) -> str:  # pragma: no cover
        raise RuntimeError("tensorrt backend not configured")

    def stream_generate(self, prompt: str, max_new_tokens: int = 64, **_kwargs) -> Iterable[str]:  # pragma: no cover
        raise RuntimeError("tensorrt backend not configured")


def load_trt_model(settings: dict) -> TRTModel:
    core = settings.get("core", {}) or {}
    device = str(core.get("trt_device") or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu"))
    engine_path = str(core.get("trt_engine_path", ""))
    tokenizer_path = core.get("trt_tokenizer_path")
    cfg = TRTConfig(engine_path=engine_path, tokenizer_path=tokenizer_path, device=device)
    return TRTModel(cfg)
