from __future__ import annotations

import json
from pathlib import Path


def test_operational_store_persists_state_events_and_runs(tmp_path: Path) -> None:
    from c3rnt2.control_plane.storage import OperationalStore

    store = OperationalStore(tmp_path / "control.sqlite3")
    store.put_state("runtime", {"mode": "primary", "updated_at": 1.0})
    store.append_event("autonomy", "global", {"id": "evt-1", "ts": 2.0, "kind": "reflection"})
    store.put_run("run-1", {"run_id": "run-1", "status": "completed", "updated_at": 3.0})
    store.append_event("training_run", "run-1", {"id": "run-evt-1", "run_id": "run-1", "ts": 4.0})

    reopened = OperationalStore(tmp_path / "control.sqlite3")

    assert reopened.get_state("runtime", {})["mode"] == "primary"
    assert reopened.list_events("autonomy", "global")[0]["id"] == "evt-1"
    assert reopened.get_run("run-1")["status"] == "completed"  # type: ignore[index]
    assert reopened.list_events("training_run", "run-1")[0]["id"] == "run-evt-1"


def test_operational_store_imports_legacy_json_and_jsonl(tmp_path: Path) -> None:
    from c3rnt2.control_plane.storage import OperationalStore

    state_path = tmp_path / "runtime_state.json"
    events_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps({"mode": "fallback", "updated_at": 5.0}), encoding="utf-8")
    events_path.write_text('{"id":"evt-legacy","ts":6.0,"kind":"legacy"}\n', encoding="utf-8")

    store = OperationalStore(tmp_path / "control.sqlite3")
    store.import_state_file("runtime", state_path, {})
    store.import_jsonl_events("autonomy", "global", events_path)

    assert store.get_state("runtime", {})["mode"] == "fallback"
    assert store.list_events("autonomy", "global")[0]["id"] == "evt-legacy"
