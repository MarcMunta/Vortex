from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


def _headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    base_url = str(args.base_url)
    token = str(args.token) if args.token else os.getenv("KLIMEAI_API_TOKEN") or None
    model = str(args.model) if args.model else None

    sess = requests.Session()

    print("[smoke] GET /healthz")
    r = sess.get(_url(base_url, "/healthz"), timeout=10)
    _assert(r.status_code == 200, f"/healthz status={r.status_code}")

    print("[smoke] GET /v1/models")
    r = sess.get(_url(base_url, "/v1/models"), headers=_headers(token), timeout=20)
    _assert(r.status_code == 200, f"/v1/models status={r.status_code} body={r.text[:200]}")
    models = r.json().get("data") or []
    _assert(isinstance(models, list), "/v1/models data must be list")
    if not model:
        model = str((models[0] or {}).get("id") if models and isinstance(models[0], dict) else "core")
    print(f"[smoke] using model={model}")

    print("[smoke] POST /v1/chat/completions (non-stream)")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "hola"}],
        "max_tokens": 16,
        "stream": False,
    }
    r = sess.post(_url(base_url, "/v1/chat/completions"), headers=_headers(token), json=payload, timeout=60)
    _assert(r.status_code == 200, f"/v1/chat/completions status={r.status_code} body={r.text[:200]}")
    data = r.json()
    text = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
    _assert(isinstance(text, str) and text, "non-stream completion missing content")

    print("[smoke] POST /v1/chat/completions (stream)")
    payload["stream"] = True
    with sess.post(_url(base_url, "/v1/chat/completions"), headers=_headers(token), json=payload, timeout=60, stream=True) as r:
        _assert(r.status_code == 200, f"stream status={r.status_code} body={r.text[:200]}")
        chunks: list[str] = []
        saw_done = False
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            payload_s = line[len("data:") :].strip()
            if payload_s == "[DONE]":
                saw_done = True
                break
            evt = json.loads(payload_s)
            choice0 = (evt.get("choices") or [{}])[0]
            delta = (choice0.get("delta") or {}).get("content") or ""
            if delta:
                chunks.append(str(delta))
        _assert(saw_done, "stream did not terminate with [DONE]")
        _assert("".join(chunks), "stream returned empty content")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

