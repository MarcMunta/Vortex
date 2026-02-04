from __future__ import annotations

import json
from pathlib import Path

from c3rnt2 import __main__ as main_mod


def test_update_bench_baseline_creates_when_missing(tmp_path: Path) -> None:
    bench_dir = tmp_path / "data" / "bench"
    baseline_path = bench_dir / "baseline.json"
    bench = {"profile": "p1", "backend": "vortex", "tokens_per_sec": 123.0}
    out = main_mod._update_bench_baseline(baseline_path, bench)
    assert out["ok"] is True
    assert out["created"] is True
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert data["p1"]["vortex"]["tokens_per_sec"] == 123.0


def test_update_bench_baseline_overwrites_existing(tmp_path: Path) -> None:
    bench_dir = tmp_path / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = bench_dir / "baseline.json"
    baseline_path.write_text(json.dumps({"p1": {"vortex": {"tokens_per_sec": 10.0}}}), encoding="utf-8")
    out = main_mod._update_bench_baseline(baseline_path, {"profile": "p1", "backend": "vortex", "tokens_per_sec": 1.0})
    assert out["ok"] is True
    assert out["updated"] is True
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert data["p1"]["vortex"]["tokens_per_sec"] == 1.0
