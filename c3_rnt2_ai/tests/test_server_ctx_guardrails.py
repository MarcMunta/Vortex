from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_server_ctx_overflow_trims_and_logs(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from c3rnt2 import server as server_mod

    class DummyModel:
        def __init__(self):
            self.tokenizer = None
            self.last_prompt = None

        def generate(self, prompt: str, **_kwargs):
            self.last_prompt = prompt
            return "ok"

    dummy = DummyModel()

    monkeypatch.setattr(server_mod, "_load_backend_model", lambda _settings, _base_dir, _backend: dummy)

    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "core": {"backend": "vortex", "hf_system_prompt": "SYS"},
        "rag": {"enabled": False},
        "decode": {"max_new_tokens": 8},
        "server": {"ctx_max_tokens": 50, "ctx_overflow_policy": "tail_keep_last_n"},
    }
    app = server_mod.create_app(settings, base_dir=tmp_path)
    client = TestClient(app)

    huge = "word " * 500
    resp = client.post("/v1/chat/completions", json={"prompt": huge, "max_tokens": 8})
    assert resp.status_code == 200
    assert dummy.last_prompt is not None
    assert len(str(dummy.last_prompt).split()) <= 50

    log_path = tmp_path / "data" / "logs" / "ctx_guard.jsonl"
    assert log_path.exists()
    rec = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    for key in ("ctx_in", "ctx_used", "ctx_dropped", "max_ctx_profile", "backend"):
        assert key in rec
    assert rec.get("max_ctx_profile") == 50
    assert rec.get("ctx_dropped") >= 1

