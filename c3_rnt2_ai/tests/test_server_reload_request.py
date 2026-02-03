from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from c3rnt2 import server as server_mod


def test_process_reload_request_updates_registry_and_removes_file(tmp_path: Path) -> None:
    base_dir = tmp_path
    adapter_dir = base_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Create reload request.
    req_path = base_dir / "data" / "state" / "reload.json"
    req_path.parent.mkdir(parents=True, exist_ok=True)
    req_path.write_text(json.dumps({"adapter_path": str(adapter_dir)}), encoding="utf-8")

    loaded = {}

    class DummyModel:
        is_hf = True

        def __init__(self):
            self.adapter_path = None

        def load_adapter(self, path: str, merge: bool = False):
            _ = merge
            loaded["path"] = path
            self.adapter_path = path

    model = DummyModel()
    app = SimpleNamespace(state=SimpleNamespace(models={"hf": model}, model=model))

    settings = {
        "core": {"hf_use_latest_adapter": True, "backend": "hf"},
        "hf_train": {"registry_dir": str(base_dir / "data" / "registry" / "hf_train")},
    }

    res = server_mod._process_reload_request(app, base_dir, settings)
    assert res.get("ok") is True
    assert res.get("processed") is True
    assert loaded.get("path") == str(adapter_dir)
    assert not req_path.exists()

