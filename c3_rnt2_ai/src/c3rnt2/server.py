from __future__ import annotations


import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from .model.core_transformer import CoreTransformer
from .model.bad_decode import _sample_logits, _sample_logits_topk, _RepetitionTracker, _NgramTracker

from .continuous.lora import LoRAConfig, inject_lora, load_lora_state, resolve_target_modules
from .continuous.registry import load_registry

def _maybe_load_adapter(model: CoreTransformer, settings: dict, base_dir: Path) -> None:
    state = load_registry(base_dir)
    if state.current_run_id:
        adapter_path = base_dir / "data" / "registry" / "adapters" / f"{state.current_run_id}.pt"
        if adapter_path.exists():
            adapter_cfg = settings.get("continuous", {}).get("adapters", {})
            lora_cfg = LoRAConfig(rank=int(adapter_cfg.get("rank", 4)), alpha=float(adapter_cfg.get("alpha", 1.0)))
            strict = bool(adapter_cfg.get("strict_target_modules", False))
            target_modules = resolve_target_modules(adapter_cfg, strict=strict)
            inject_lora(model, lora_cfg, target_modules=target_modules)
            load_lora_state(model, adapter_path)



def _build_prompt(messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if role == "system":
            parts.append(f"### System:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
        else:
            parts.append(f"### User:\n{content}")
    parts.append("### Assistant:\n")
    return "\n".join(parts).strip()


def _stream_generate(
    model: CoreTransformer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram: int,
    penalty_window: int,
    top_p_min_k: int,
    top_p_max_k: int,
) -> Iterable[str]:
    ids, _total = model.encode_prompt(prompt)
    if not ids:
        ids = [0]
    rep_tracker = _RepetitionTracker(penalty_window)
    ngram_tracker = _NgramTracker(no_repeat_ngram)
    for tok in ids:
        rep_tracker.add(tok)
        ngram_tracker.add(tok)
    prev_text = model.decode_ids(ids, total_len=None)

    stream_topk_cfg = getattr(model, "runtime_cfg", {}).get("paged_lm_head_stream_topk", False)
    stream_topk = bool(stream_topk_cfg)
    top_k = int(stream_topk_cfg) if isinstance(stream_topk_cfg, int) else 64

    model.reset_state()
    _last_logits, state = model.init_state(prompt_ids=ids, return_logits=True, write_memory=True)
    last_token = ids[-1]

    for _ in range(max_new_tokens):
        if stream_topk and hasattr(model, "step_topk"):
            values, indices, state = model.step_topk(last_token, state, top_k=top_k, write_memory=True)
            next_tok_t = _sample_logits_topk(
                values,
                indices,
                temperature,
                top_p,
                repetition_penalty,
                rep_tracker,
                ngram_tracker,
                top_p_min_k=top_p_min_k,
                top_p_max_k=top_p_max_k,
            )
        else:
            logits, state = model.step(last_token, state, write_memory=True)
            next_tok_t = _sample_logits(
                logits,
                temperature,
                top_p,
                repetition_penalty,
                rep_tracker,
                ngram_tracker,
                top_p_min_k=top_p_min_k,
                top_p_max_k=top_p_max_k,
            )
        token_id = int(next_tok_t.item())
        ids.append(token_id)
        rep_tracker.add(token_id)
        ngram_tracker.add(token_id)
        last_token = token_id
        text = model.decode_ids(ids, total_len=None)
        delta = text[len(prev_text):]
        prev_text = text
        if delta:
            yield delta


def _resolve_decode_args(settings: dict, payload: dict) -> dict[str, Any]:
    decode_cfg = settings.get("decode", {}) or {}
    bad_cfg = settings.get("bad", {}) or {}
    max_tokens = payload.get("max_tokens") or payload.get("max_new_tokens") or decode_cfg.get("max_new_tokens", 64)
    return {
        "max_new_tokens": int(max_tokens),
        "temperature": float(payload.get("temperature", decode_cfg.get("temperature", 1.0))),
        "top_p": float(payload.get("top_p", decode_cfg.get("top_p", 1.0))),
        "repetition_penalty": float(payload.get("repetition_penalty", decode_cfg.get("repetition_penalty", 1.0))),
        "no_repeat_ngram": int(payload.get("no_repeat_ngram", decode_cfg.get("no_repeat_ngram", 0))),
        "penalty_window": int(payload.get("penalty_window", bad_cfg.get("penalty_window", 512))),
        "top_p_min_k": int(payload.get("top_p_min_k", bad_cfg.get("top_p_min_k", 128))),
        "top_p_max_k": int(payload.get("top_p_max_k", bad_cfg.get("top_p_max_k", 512))),
    }


def create_app(settings: dict, base_dir: Path) -> "FastAPI":
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"FastAPI not available: {exc}")

    app = FastAPI()
    model = CoreTransformer.from_settings(settings)
    _maybe_load_adapter(model, settings, base_dir)
    model_lock = threading.Lock()
    app.state.model = model
    app.state.settings = settings
    app.state.model_lock = model_lock

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        messages = payload.get("messages") or []
        if not messages and payload.get("prompt"):
            messages = [{"role": "user", "content": payload.get("prompt")}]
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")
        prompt = _build_prompt(messages)
        stream = bool(payload.get("stream", False))
        decode_args = _resolve_decode_args(settings, payload)
        created = int(time.time())
        resp_id = f"chatcmpl-{created}"

        if not stream:
            with model_lock:
                text = model.generate(prompt, max_new_tokens=decode_args["max_new_tokens"], temperature=decode_args["temperature"], top_p=decode_args["top_p"], repetition_penalty=decode_args["repetition_penalty"], no_repeat_ngram=decode_args["no_repeat_ngram"])
            data = {
                "id": resp_id,
                "object": "chat.completion",
                "created": created,
                "model": "vortex-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
            return JSONResponse(content=data)

        def event_stream() -> Iterable[str]:
            header = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "vortex-x",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(header)}\n\n"
            with model_lock:
                for delta in _stream_generate(model, prompt, **decode_args):
                    chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "vortex-x",
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "vortex-x",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def run_server(settings: dict, base_dir: Path, host: str = "0.0.0.0", port: int = 8000) -> None:
    try:
        import uvicorn  # type: ignore
        app = create_app(settings, base_dir=base_dir)
    except Exception:
        _run_basic_server(settings, base_dir, host, port)
        return
    uvicorn.run(app, host=host, port=port, log_level="info")



def _run_basic_server(settings: dict, base_dir: Path, host: str, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model = CoreTransformer.from_settings(settings)
    _maybe_load_adapter(model, settings, base_dir)
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: N802
            return

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            messages = payload.get("messages") or []
            if not messages and payload.get("prompt"):
                messages = [{"role": "user", "content": payload.get("prompt")}]
            if not messages:
                self.send_response(400)
                self.end_headers()
                return
            prompt = _build_prompt(messages)
            stream = bool(payload.get("stream", False))
            decode_args = _resolve_decode_args(settings, payload)
            created = int(time.time())
            resp_id = f"chatcmpl-{created}"

            if not stream:
                with model_lock:
                    text = model.generate(prompt, max_new_tokens=decode_args["max_new_tokens"], temperature=decode_args["temperature"], top_p=decode_args["top_p"], repetition_penalty=decode_args["repetition_penalty"], no_repeat_ngram=decode_args["no_repeat_ngram"])
                data = {
                    "id": resp_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": "vortex-x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                }
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            header = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "vortex-x",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(header)}\n\n".encode("utf-8"))
            self.wfile.flush()
            with model_lock:
                for delta in _stream_generate(model, prompt, **decode_args):
                    chunk = {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "vortex-x",
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            done = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "vortex-x",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer((host, port), Handler)
    print({"ok": True, "mode": "basic", "host": host, "port": port})
    server.serve_forever()

