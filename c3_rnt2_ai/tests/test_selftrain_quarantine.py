from __future__ import annotations

import json
from pathlib import Path

from c3rnt2.continuous.promotion import promote_quarantine_run, quarantine_run_dir


def test_quarantine_requires_approval(tmp_path: Path) -> None:
    run_id = "run1"
    qdir = quarantine_run_dir(tmp_path, run_id)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "adapter.pt").write_bytes(b"abc")
    (qdir / "manifest.json").write_text(json.dumps({"run_id": run_id, "passed_eval": True}), encoding="utf-8")

    res = promote_quarantine_run(tmp_path, run_id=run_id, require_approval=True)
    assert res.ok
    assert not res.promoted
    assert res.reason == "approval_missing"
    assert not (tmp_path / "data" / "continuous" / "promoted" / run_id).exists()

    (qdir / "APPROVE.txt").write_text("ok", encoding="utf-8")
    res2 = promote_quarantine_run(tmp_path, run_id=run_id, require_approval=True)
    assert res2.ok
    assert res2.promoted
    current = tmp_path / "data" / "continuous" / "current_adapter.json"
    assert current.exists()
    payload = json.loads(current.read_text(encoding="utf-8"))
    assert payload["current_run_id"] == run_id


def test_quarantine_requires_passed_eval(tmp_path: Path) -> None:
    run_id = "run2"
    qdir = quarantine_run_dir(tmp_path, run_id)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "adapter.pt").write_bytes(b"abc")
    (qdir / "manifest.json").write_text(json.dumps({"run_id": run_id, "passed_eval": False}), encoding="utf-8")
    (qdir / "APPROVE.txt").write_text("ok", encoding="utf-8")

    res = promote_quarantine_run(tmp_path, run_id=run_id, require_approval=True)
    assert res.ok
    assert not res.promoted
    assert res.reason == "passed_eval_false"
