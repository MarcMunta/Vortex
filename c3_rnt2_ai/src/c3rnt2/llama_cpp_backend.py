from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .prompting.chat_format import build_chat_prompt


def _log_infer_stats(base_dir: Path, payload: dict) -> None:
    log_dir = base_dir / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llama_cpp_infer.jsonl"
    meta_path = log_dir / "llama_cpp_infer_meta.json"
    payload = dict(payload)
    payload.setdefault("ts", time.time())
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    try:
        meta_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass


def _filter_kwargs(fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except Exception:
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _coerce_int(value: object | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_bool(value: object | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class LlamaCppConfig:
    model_path: Path
    n_gpu_layers: int | None
    n_ctx: int | None
    n_threads: int | None
    n_batch: int | None
    flash_attn: bool
    chat_format: str | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None


class LlamaCppModel:
    is_llama_cpp = True
    is_hf = False

    def __init__(self, cfg: LlamaCppConfig):
        try:
            import llama_cpp  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"llama_cpp not available: {exc}")
        self.cfg = cfg
        model_kwargs: dict[str, Any] = {
            "model_path": str(cfg.model_path),
            "n_gpu_layers": cfg.n_gpu_layers,
            "n_ctx": cfg.n_ctx,
            "n_threads": cfg.n_threads,
            "n_batch": cfg.n_batch,
            "flash_attn": bool(cfg.flash_attn),
            "chat_format": cfg.chat_format,
            "rope_freq_base": cfg.rope_freq_base,
            "rope_freq_scale": cfg.rope_freq_scale,
            "verbose": False,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        model_kwargs = _filter_kwargs(llama_cpp.Llama.__init__, model_kwargs)
        self.model = llama_cpp.Llama(**model_kwargs)
        self.tokenizer = None
        self.base_dir = Path(".")

    def _prepare_prompt(self, prompt: str | None, messages: list[dict] | None, system: str | None) -> str:
        if messages is not None:
            return build_chat_prompt(messages, backend="llama_cpp", tokenizer=None, default_system=system)
        return prompt or ""

    def _prepare_messages(self, messages: list[dict] | None, system: str | None) -> list[dict] | None:
        if messages is None:
            return None
        normalized = [dict(item) for item in messages if isinstance(item, dict)]
        if system and not any(str(item.get("role") or "").lower() == "system" for item in normalized):
            normalized.insert(0, {"role": "system", "content": str(system)})
        return normalized

    def encode_prompt(self, prompt: str):
        try:
            if hasattr(self.model, "tokenize"):
                try:
                    ids = self.model.tokenize(prompt.encode("utf-8"))  # type: ignore[attr-defined]
                except TypeError:
                    ids = self.model.tokenize(prompt.encode("utf-8"), add_bos=False)  # type: ignore[attr-defined]
                ids = list(ids) if isinstance(ids, (list, tuple)) else []
                return ids, int(len(ids))
        except Exception:
            pass
        est = len((prompt or "").split())
        return [], int(est)

    def decode_ids(self, ids: list[int], total_len: int | None = None) -> str:
        _ = total_len
        if not ids:
            return ""
        try:
            if hasattr(self.model, "detokenize"):
                raw = self.model.detokenize(ids)  # type: ignore[attr-defined]
                if isinstance(raw, (bytes, bytearray)):
                    return raw.decode("utf-8", errors="ignore")
                return str(raw)
        except Exception:
            return ""
        return ""

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
        _ = no_repeat_ngram
        chat_messages = self._prepare_messages(messages, system)
        prompt_text = self._prepare_prompt(prompt, messages, system)
        start = time.time()
        out = None
        max_new = max(1, int(max_new_tokens))
        for _attempt in range(2):
            try:
                if chat_messages is not None and hasattr(self.model, "create_chat_completion"):
                    kwargs = {
                        "messages": chat_messages,
                        "max_tokens": max_new,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "repeat_penalty": float(repetition_penalty),
                        "stream": False,
                    }
                    kwargs = _filter_kwargs(getattr(self.model, "create_chat_completion"), kwargs)
                    out = self.model.create_chat_completion(**kwargs)
                else:
                    kwargs = {
                        "prompt": prompt_text,
                        "max_tokens": max_new,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "repeat_penalty": float(repetition_penalty),
                        "stream": False,
                    }
                    kwargs = _filter_kwargs(getattr(self.model, "create_completion"), kwargs)
                    out = self.model.create_completion(**kwargs)
                break
            except Exception as exc:
                if _attempt == 0 and max_new > 1:
                    max_new = max(1, max_new // 2)
                    out = None
                    continue
                raise

        text = ""
        usage = {}
        if isinstance(out, dict):
            usage = out.get("usage", {}) if isinstance(out.get("usage"), dict) else {}
            try:
                choices = out.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0] or {}
                    message = first.get("message") if isinstance(first, dict) else None
                    if isinstance(message, dict) and message.get("content") is not None:
                        text = str(message.get("content") or "")
                    else:
                        text = str(first.get("text", ""))
            except Exception:
                text = ""

        elapsed = max(1e-6, time.time() - start)
        completion_tokens = usage.get("completion_tokens")
        try:
            tokens = int(completion_tokens) if completion_tokens is not None else None
        except Exception:
            tokens = None
        if tokens is None:
            try:
                tokens = len(self.encode_prompt(text)[0])
            except Exception:
                tokens = len(text.split())

        _log_infer_stats(
            Path(getattr(self, "base_dir", Path("."))),
            {
                "tokens": int(tokens),
                "tokens_per_sec": float(tokens) / elapsed,
                "backend": "llama_cpp",
                "model_path": str(self.cfg.model_path),
                "n_gpu_layers": self.cfg.n_gpu_layers,
                "ctx": self.cfg.n_ctx,
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
        **_kwargs,
    ) -> Iterable[str]:
        _ = no_repeat_ngram
        chat_messages = self._prepare_messages(messages, system)
        prompt_text = self._prepare_prompt(prompt, messages, system)
        start = time.time()
        max_new = max(1, int(max_new_tokens))
        chunks: list[str] = []
        usage_tokens: int | None = None
        try:
            if chat_messages is not None and hasattr(self.model, "create_chat_completion"):
                kwargs = {
                    "messages": chat_messages,
                    "max_tokens": max_new,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "repeat_penalty": float(repetition_penalty),
                    "stream": True,
                }
                stream_fn = getattr(self.model, "create_chat_completion")
            else:
                kwargs = {
                    "prompt": prompt_text,
                    "max_tokens": max_new,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "repeat_penalty": float(repetition_penalty),
                    "stream": True,
                }
                stream_fn = getattr(self.model, "create_completion")
            kwargs = _filter_kwargs(stream_fn, kwargs)
            for part in stream_fn(**kwargs):
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("usage"), dict) and part["usage"].get("completion_tokens") is not None:
                    try:
                        usage_tokens = int(part["usage"]["completion_tokens"])
                    except Exception:
                        usage_tokens = None
                try:
                    choices = part.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0] or {}
                        delta_obj = first.get("delta") if isinstance(first, dict) else None
                        if isinstance(delta_obj, dict) and delta_obj.get("content") is not None:
                            delta = str(delta_obj.get("content") or "")
                        else:
                            delta = str(first.get("text", ""))
                    else:
                        delta = ""
                except Exception:
                    delta = ""
                if delta:
                    chunks.append(delta)
                    yield delta
        finally:
            elapsed = max(1e-6, time.time() - start)
            if usage_tokens is not None:
                tokens = usage_tokens
            else:
                try:
                    tokens = len(self.encode_prompt("".join(chunks))[0])
                except Exception:
                    tokens = len("".join(chunks).split())
            _log_infer_stats(
                Path(getattr(self, "base_dir", Path("."))),
                {
                    "tokens": int(tokens),
                    "tokens_per_sec": float(tokens) / elapsed,
                    "backend": "llama_cpp",
                    "model_path": str(self.cfg.model_path),
                    "n_gpu_layers": self.cfg.n_gpu_layers,
                    "ctx": self.cfg.n_ctx,
                    "stream": True,
                },
            )


def load_llama_cpp_model(settings: dict, *, base_dir: Path | None = None) -> LlamaCppModel:
    base_dir = Path(base_dir or ".").resolve()
    core = settings.get("core", {}) or {}
    model_path = core.get("llama_cpp_model_path")
    if not model_path:
        raise ValueError("core.llama_cpp_model_path is required for llama_cpp backend")
    path = Path(str(model_path))
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"GGUF model not found: {path}")

    cfg = LlamaCppConfig(
        model_path=path,
        n_gpu_layers=_coerce_int(core.get("llama_cpp_n_gpu_layers"), default=None),
        n_ctx=_coerce_int(core.get("llama_cpp_ctx"), default=None),
        n_threads=_coerce_int(core.get("llama_cpp_threads"), default=None),
        n_batch=_coerce_int(core.get("llama_cpp_batch"), default=None),
        flash_attn=_coerce_bool(core.get("llama_cpp_flash_attn"), default=False),
        chat_format=str(core.get("llama_cpp_chat_format") or "").strip() or None,
        rope_freq_base=float(core["llama_cpp_rope_freq_base"]) if core.get("llama_cpp_rope_freq_base") is not None else None,
        rope_freq_scale=float(core["llama_cpp_rope_freq_scale"]) if core.get("llama_cpp_rope_freq_scale") is not None else None,
    )
    model = LlamaCppModel(cfg)
    try:
        model.base_dir = base_dir
    except Exception:
        pass
    return model
