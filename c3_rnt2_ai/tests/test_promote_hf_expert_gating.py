from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.continuous.promotion import promote_hf_expert


def test_promote_hf_expert_blocks_on_bench_and_logs_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    # Quarantine layout: <run_dir>/adapter + manifest.json
    run_dir = tmp_path / "data" / "experts_hf" / "quarantine" / "code" / "run1"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "MOCK.txt").write_text("x", encoding="utf-8")

    manifest = {
        "version": 1,
        "kind": "hf_expert",
        "profile": "rtx4080_16gb_120b_like",
        "domain": "code",
        "run_id": "run1",
        "passed_eval": True,
        "bench": {"tokens_per_sec": 1.0, "vram_peak_mb": 99999.0, "ctx": 2048},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")

    # Manual approval file required by 120B-like profile.
    approve = tmp_path / "data" / "APPROVE_PROMOTION"
    approve.parent.mkdir(parents=True, exist_ok=True)
    approve.write_text("ok\n", encoding="utf-8")

    # Baseline required for 120B-like HF expert promotion.
    bench_dir = tmp_path / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "baseline.json").write_text('{"rtx4080_16gb_120b_like": {"hf": {"tokens_per_sec": 10.0}}}', encoding="utf-8")

    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "bench_thresholds": {"min_tokens_per_sec": 10.0, "max_regression": 0.15, "max_vram_peak_mb": 15500, "required_ctx": 4096},
    }
    out = promote_hf_expert(run_dir, tmp_path / "data" / "experts_hf" / "registry", settings)
    assert out["ok"] is True
    assert out["promoted"] is False
    assert str(out.get("reason") or "").startswith("bench_failed:")

    # Fail reason is logged.
    log_path = tmp_path / "data" / "logs" / "promotions.jsonl"
    assert log_path.exists()
    rec = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec.get("promote_ok") is False
    assert "below_min_tokens_per_sec" in (rec.get("reason") or "")


def test_promote_hf_expert_blocks_when_baseline_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "data" / "experts_hf" / "quarantine" / "code" / "run_ok"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "MOCK.txt").write_text("x", encoding="utf-8")

    manifest = {
        "version": 1,
        "kind": "hf_expert",
        "profile": "rtx4080_16gb_120b_like",
        "domain": "code",
        "run_id": "run_ok",
        "passed_eval": True,
        "bench": {"tokens_per_sec": 20.0, "vram_peak_mb": 1000.0, "ctx": 4096},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")

    approve = tmp_path / "data" / "APPROVE_PROMOTION"
    approve.parent.mkdir(parents=True, exist_ok=True)
    approve.write_text("ok\n", encoding="utf-8")

    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "bench_thresholds": {"min_tokens_per_sec": 10.0, "max_regression": 0.15, "max_vram_peak_mb": 15500, "required_ctx": 4096},
    }
    out = promote_hf_expert(run_dir, tmp_path / "data" / "experts_hf" / "registry", settings)
    assert out["ok"] is True
    assert out["promoted"] is False
    assert out.get("reason") == "baseline_missing"

    log_path = tmp_path / "data" / "logs" / "promotions.jsonl"
    assert log_path.exists()
    rec = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec.get("promote_ok") is False
    assert rec.get("reason") == "baseline_missing"
