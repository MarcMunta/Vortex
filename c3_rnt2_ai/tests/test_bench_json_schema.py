from __future__ import annotations

import json
from pathlib import Path

import c3rnt2.bench as bench_mod
from c3rnt2.bench import BenchArgs, run_bench


class _DummyModel:
    runtime_cfg = {"paged_lm_head": False}
    lm_head = None

    def encode_prompt(self, text: str):  # type: ignore[no-untyped-def]
        ids = list(range(len(text.encode("utf-8"))))
        return ids, len(ids)

    def generate(self, _prompt: str, *, max_new_tokens: int):  # type: ignore[no-untyped-def]
        _ = max_new_tokens
        return "ok"


def test_bench_json_schema_minimal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bench_mod, "load_inference_model", lambda _settings: _DummyModel())
    settings = {"_profile": "p", "core": {"backend": "vortex"}}
    out_path = tmp_path / "data" / "bench" / "last.json"
    report = run_bench(
        settings,
        base_dir=tmp_path,
        args=BenchArgs(
            profile="p",
            prompt="hello",
            prompt_file=None,
            ctx=None,
            max_new=4,
            warmup=1,
            repeat=2,
            seed=0,
            json_out=out_path,
            jsonl_out=None,
        ),
    )
    assert report["ok"] is True
    for key in (
        "profile",
        "backend",
        "ctx_len_prompt",
        "ctx_len_total",
        "max_new_tokens",
        "tokens_per_sec",
        "tokens_per_sec_warmup",
        "tokens_per_sec_steady",
        "latency_ms_total",
        "latency_ms_per_token",
        "vram_peak_mb",
        "vram_peak_mb_allocated",
        "vram_peak_mb_reserved",
        "ram_rss_mb",
        "cache_hit_rate",
        "bytes_prefetched",
        "page_faults",
    ):
        assert key in report

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "p"
