import json
from pathlib import Path

from c3rnt2.training.train_router import train_router
from c3rnt2.runtime.router import load_router


def test_router_training_and_decision(tmp_path: Path):
    data = tmp_path / "router_events.jsonl"
    events = [
        {
            "request_id": "1",
            "backend": "core",
            "stream_topk": True,
            "prompt_tokens": 64,
            "max_new_tokens": 64,
            "tok_per_s": 200.0,
            "vram_peak_mb": 2000.0,
            "verify_accept_rate": 0.7,
            "error_rate": 0.0,
        },
        {
            "request_id": "1",
            "backend": "hf",
            "stream_topk": False,
            "prompt_tokens": 64,
            "max_new_tokens": 64,
            "tok_per_s": 50.0,
            "vram_peak_mb": 8000.0,
            "verify_accept_rate": 0.7,
            "error_rate": 0.0,
        },
    ]
    data.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    out = tmp_path / "router.pt"
    result = train_router(data, out, weights={"speed": 1.0, "vram": 0.01, "error": 1.0, "quality": 1.0}, epochs=50, hidden_size=8, seed=123)
    assert result["ok"]
    router = load_router(out, out.with_suffix(".json"))
    assert router is not None
    decision = router.decide({
        "prompt_tokens": 64,
        "max_new_tokens": 64,
        "vram_budget": 4000,
        "tok_per_s": 100,
        "vram_peak_mb": 2000,
        "verify_accept_rate": 0.7,
        "error_rate": 0.0,
    })
    assert decision.backend == "core"
