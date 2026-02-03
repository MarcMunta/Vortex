from __future__ import annotations

from pathlib import Path

from c3rnt2.adapters.registry import AdapterRegistry
from c3rnt2.adapters.router import AdapterRouter
from c3rnt2.server import _ensure_hf_adapter, _select_hf_adapter_for_request


def test_adapter_registry_resolves_relative_paths(tmp_path: Path) -> None:
    settings = {
        "adapters": {
            "enabled": True,
            "paths": {"programming": "data/adapters/prog"},
            "max_loaded": 2,
            "default": "programming",
        }
    }
    reg = AdapterRegistry.from_settings(settings, base_dir=tmp_path)
    assert reg.enabled is True
    assert reg.max_loaded == 2
    assert reg.default == "programming"
    assert reg.get_path("programming") == str(tmp_path / "data" / "adapters" / "prog")


def test_adapter_router_keyword_map_selects_adapter() -> None:
    router = AdapterRouter(mode="keyword_map", keyword_map={"python": "programming"}, default_adapter="general")
    decision = router.select("write python code", ["general", "programming"])
    assert decision.selected_adapter == "programming"
    assert decision.reason.startswith("keyword:")


def test_select_hf_adapter_for_request_rejects_unknown_adapter() -> None:
    reg = AdapterRegistry(enabled=True, paths={"a": "/tmp/a"}, max_loaded=0, default=None)
    router = AdapterRouter(mode="keyword_map", keyword_map={}, default_adapter=None)
    out = _select_hf_adapter_for_request({"adapter": "missing"}, "prompt", reg, router)
    assert out["ok"] is False
    assert out["error"] == "adapter_not_found"


def test_ensure_hf_adapter_calls_model(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    reg = AdapterRegistry(enabled=True, paths={"programming": str(adapter_dir)}, max_loaded=1, default=None)

    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def add_adapter(self, name: str, path: str) -> None:
            self.calls.append(("add", name, path))

        def set_adapter(self, name: str) -> None:
            self.calls.append(("set", name))

    model = DummyModel()
    out = _ensure_hf_adapter(model, reg, "programming")
    assert out["ok"] is True
    assert ("set", "programming") in model.calls

