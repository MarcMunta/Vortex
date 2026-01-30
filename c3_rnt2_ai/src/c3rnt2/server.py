from __future__ import annotations

import json
import threading
import time
from copy import deepcopy

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from pathlib import Path
from typing import Any, Dict, Iterable

from .model.core_transformer import CoreTransformer
from .model_loader import load_inference_model
from .prompting.chat_format import build_chat_prompt
from .model.bad_decode import _sample_logits, _sample_logits_topk, _RepetitionTracker, _NgramTracker
from .continuous.lora import LoRAConfig, inject_lora, load_lora_state, resolve_target_modules
from .continuous.registry import load_registry
from .runtime.router import build_features, load_router, log_router_event


def _maybe_load_adapter(model: CoreTransformer, settings: dict, base_dir: Path) -> None:
    if not hasattr(model, "blocks"):
        return
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


def _load_backend_model(settings: dict, base_dir: Path, backend: str):
    local = deepcopy(settings)
    core = local.get("core", {}) or {}
    if backend == "hf":
        core["backend"] = "hf"
    else:
        core["backend"] = "vortex"
    local["core"] = core
    model = load_inference_model(local)
    _maybe_load_adapter(model, local, base_dir)
    return model


def _maybe_set_stream_topk(model: CoreTransformer, enabled: bool, top_k: int | None = None):
    if not hasattr(model, "runtime_cfg"):
        return None
    runtime = model.runtime_cfg
    if runtime is None:
        return None
    original = runtime.get("paged_lm_head_stream_topk", False)
    if enabled:
        if top_k is None:
            top_k = int(original) if isinstance(original, int) and original > 0 else 64
        runtime["paged_lm_head_stream_topk"] = int(top_k)
    else:
        runtime["paged_lm_head_stream_topk"] = False
    return original



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
    router_cfg = settings.get("core", {}).get("router", settings.get("router", {})) or {}
    router_enabled = bool(router_cfg.get("enabled", False))
    router = None
    router_path = Path(router_cfg.get("path", "data/runs/router.pt"))
    if router_enabled:
        router = load_router(router_path, router_path.with_suffix(".json"))

    core_backend = str(settings.get("core", {}).get("backend", "vortex")).lower()
    default_backend_label = "hf" if core_backend == "hf" else "core"
    model = _load_backend_model(settings, base_dir, default_backend_label)
    models = {default_backend_label: model}
    if router_enabled and bool(router_cfg.get("multi_backend", False)):
        for backend in ("core", "hf"):
            if backend in models:
                continue
            try:
                models[backend] = _load_backend_model(settings, base_dir, backend)
            except Exception:
                continue

    model_lock = threading.Lock()
    app.state.model = model
    app.state.models = models
    app.state.settings = settings
    app.state.model_lock = model_lock
    app.state.router = router
    app.state.router_cfg = router_cfg

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        messages = payload.get("messages") or []
        if not messages and payload.get("prompt"):
            messages = [{"role": "user", "content": payload.get("prompt")}]
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")
        backend_cfg = settings.get("core", {}).get("backend", "vortex")
        default_system = settings.get("core", {}).get("hf_system_prompt", "You are a helpful coding assistant.")
        prompt = build_chat_prompt(messages, backend_cfg, tokenizer=getattr(model, "tokenizer", None), default_system=default_system)
        stream = bool(payload.get("stream", False))
        decode_args = _resolve_decode_args(settings, payload)
        created = int(time.time())
        resp_id = f"chatcmpl-{created}"

        chosen_backend = default_backend_label
        decision = None
        if router is not None:
            feats = build_features(prompt, decode_args["max_new_tokens"], settings)
            decision = router.decide(feats)
            chosen_backend = decision.backend
        selected_model = models.get(chosen_backend, model)
        stream_topk_override = None
        if decision is not None and hasattr(selected_model, "runtime_cfg"):
            stream_topk_override = _maybe_set_stream_topk(
                selected_model,
                enabled=bool(decision.stream_topk),
                top_k=int(router_cfg.get("stream_topk_k", 64)),
            )

        if not stream:
            start = time.time()
            with model_lock:
                if hasattr(selected_model, "generate"):
                    if hasattr(selected_model, "blocks"):
                        text, stats = selected_model.generate(
                            prompt,
                            max_new_tokens=decode_args["max_new_tokens"],
                            temperature=decode_args["temperature"],
                            top_p=decode_args["top_p"],
                            repetition_penalty=decode_args["repetition_penalty"],
                            no_repeat_ngram=decode_args["no_repeat_ngram"],
                            return_stats=True,
                        )
                    else:
                        text = selected_model.generate(
                            prompt,
                            max_new_tokens=decode_args["max_new_tokens"],
                            temperature=decode_args["temperature"],
                            top_p=decode_args["top_p"],
                            repetition_penalty=decode_args["repetition_penalty"],
                            no_repeat_ngram=decode_args["no_repeat_ngram"],
                        )
                        stats = None
                else:
                    text = ""
                    stats = None
            elapsed = max(1e-6, time.time() - start)
            vram_peak = None
            if torch is not None and torch.cuda.is_available():
                try:
                    vram_peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                except Exception:
                    vram_peak = None
            mem_cost = 0.0
            if hasattr(selected_model, "blocks"):
                try:
                    mem_cost = float(sum(block.lava.stats.reads + block.lava.stats.writes for block in selected_model.blocks))
                except Exception:
                    mem_cost = 0.0
            if router is not None:
                verify_rate = 0.0
                if stats is not None:
                    verify_rate = float(stats.accepted) / max(1, int(stats.proposed))
                log_router_event(
                    base_dir,
                    {
                        "request_id": resp_id,
                        "backend": chosen_backend,
                        "stream_topk": bool(decision.stream_topk) if decision else False,
                        "prompt_tokens": len(prompt.split()),
                        "max_new_tokens": decode_args["max_new_tokens"],
                        "tok_per_s": float(decode_args["max_new_tokens"]) / elapsed,
                        "verify_accept_rate": verify_rate,
                        "error_rate": 0.0,
                        "vram_peak_mb": vram_peak or 0.0,
                        "mem_cost_estimate": mem_cost,
                    },
                )
            if stream_topk_override is not None:
                _maybe_set_stream_topk(selected_model, enabled=bool(stream_topk_override), top_k=int(stream_topk_override) if stream_topk_override else None)
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
            chunks: list[str] = []
            start = time.time()
            with model_lock:
                if hasattr(selected_model, "stream_generate"):
                    for delta in selected_model.stream_generate(prompt, **decode_args):
                        chunks.append(delta)
                        chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "vortex-x",
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    for delta in _stream_generate(selected_model, prompt, **decode_args):
                        chunks.append(delta)
                        chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "vortex-x",
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
            elapsed = max(1e-6, time.time() - start)
            vram_peak = None
            if torch is not None and torch.cuda.is_available():
                try:
                    vram_peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                except Exception:
                    vram_peak = None
            mem_cost = 0.0
            if hasattr(selected_model, "blocks"):
                try:
                    mem_cost = float(sum(block.lava.stats.reads + block.lava.stats.writes for block in selected_model.blocks))
                except Exception:
                    mem_cost = 0.0
            if router is not None:
                full_text = "".join(chunks)
                token_count = len(full_text.split())
                log_router_event(
                    base_dir,
                    {
                        "request_id": resp_id,
                        "backend": chosen_backend,
                        "stream_topk": bool(decision.stream_topk) if decision else False,
                        "prompt_tokens": len(prompt.split()),
                        "max_new_tokens": decode_args["max_new_tokens"],
                        "tok_per_s": float(token_count) / elapsed,
                        "verify_accept_rate": 0.0,
                        "error_rate": 0.0,
                        "vram_peak_mb": vram_peak or 0.0,
                        "mem_cost_estimate": mem_cost,
                    },
                )
            if stream_topk_override is not None:
                _maybe_set_stream_topk(selected_model, enabled=bool(stream_topk_override), top_k=int(stream_topk_override) if stream_topk_override else None)
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

    model = load_inference_model(settings)
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
            backend = settings.get("core", {}).get("backend", "vortex")
            default_system = settings.get("core", {}).get("hf_system_prompt", "You are a helpful coding assistant.")
            prompt = build_chat_prompt(messages, backend, tokenizer=getattr(model, "tokenizer", None), default_system=default_system)
            stream = bool(payload.get("stream", False))
            decode_args = _resolve_decode_args(settings, payload)
            created = int(time.time())
            resp_id = f"chatcmpl-{created}"

            if not stream:
                with model_lock:
                    text = model.generate(
                        prompt,
                        max_new_tokens=decode_args["max_new_tokens"],
                        temperature=decode_args["temperature"],
                        top_p=decode_args["top_p"],
                        repetition_penalty=decode_args["repetition_penalty"],
                        no_repeat_ngram=decode_args["no_repeat_ngram"],
                    )
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
                if hasattr(model, "stream_generate"):
                    for delta in model.stream_generate(prompt, **decode_args):
                        chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "vortex-x",
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                else:
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
