from __future__ import annotations

import json
from pathlib import Path

from c3rnt2 import __main__ as main_mod


def test_update_bench_baseline_creates_when_missing(tmp_path: Path) -> None:
    bench_dir = tmp_path / "data" / "bench"
    bench = {"profile": "p1", "backend": "vortex", "tokens_per_sec": 123.0}
    out = main_mod._update_bench_baseline(bench_dir, bench)
    assert out["ok"] is True
    assert out["baseline_created"] is True
    baseline_path = bench_dir / "baseline.json"
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert data["p1"]["vortex"]["tokens_per_sec"] == 123.0


def test_update_bench_baseline_no_overwrite(tmp_path: Path) -> None:
    bench_dir = tmp_path / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "baseline.json").write_text(json.dumps({"p1": {"vortex": {"tokens_per_sec": 10.0}}}), encoding="utf-8")
    out = main_mod._update_bench_baseline(bench_dir, {"profile": "p1", "backend": "vortex", "tokens_per_sec": 1.0})
    assert out["ok"] is True
    assert out["baseline_created"] is False
    data = json.loads((bench_dir / "baseline.json").read_text(encoding="utf-8"))
    assert data["p1"]["vortex"]["tokens_per_sec"] == 10.0

