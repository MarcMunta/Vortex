from __future__ import annotations

from c3rnt2.promotion.gating import bench_gate, compare_to_baseline


def test_bench_gate_fails_on_min_tps() -> None:
    verdict = bench_gate(
        {"tokens_per_sec": 5.0},
        baseline={"tokens_per_sec": 10.0},
        thresholds={"min_tokens_per_sec": 6.0, "max_regression": 1.0},
    )
    assert verdict["ok"] is False
    assert "below_min_tokens_per_sec" in (verdict["reason"] or "")


def test_bench_gate_fails_on_regression() -> None:
    verdict = bench_gate(
        {"tokens_per_sec": 8.0},
        baseline={"tokens_per_sec": 10.0},
        thresholds={"min_tokens_per_sec": 0.0, "max_regression": 0.1},
    )
    assert verdict["ok"] is False
    assert "regression_exceeded" in (verdict["reason"] or "")


def test_bench_gate_fails_on_vram_or_ctx() -> None:
    verdict = bench_gate(
        {"tokens_per_sec": 100.0, "vram_peak_mb": 2000.0, "ctx": 2048},
        baseline=None,
        thresholds={"min_tokens_per_sec": 0.0, "max_regression": 1.0, "max_vram_peak_mb": 1000.0, "required_ctx": 4096},
    )
    assert verdict["ok"] is False
    assert "vram_peak_exceeded" in (verdict["reason"] or "")
    assert "ctx_too_small" in (verdict["reason"] or "")


def test_compare_to_baseline_returns_ok_reason_tuple() -> None:
    ok, reason = compare_to_baseline(
        {"tokens_per_sec": 10.0},
        baseline={"tokens_per_sec": 10.0},
        thresholds={"min_tokens_per_sec": 1.0, "max_regression": 0.1},
    )
    assert ok is True
    assert reason == ""
