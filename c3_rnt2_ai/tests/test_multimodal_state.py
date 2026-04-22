from __future__ import annotations

from pathlib import Path

from c3rnt2.multimodal import ObsidianSyncService, PanelRegistry, SpatialStateStore
from c3rnt2.multimodal.voice_models import extract_voice_intent


def _settings(tmp_path: Path) -> dict:
    return {
        "multimodal_memory": {"state_path": str(tmp_path / "state" / "spatial.json")},
        "obsidian": {
            "enabled": True,
            "vault_path": str(tmp_path / "vault"),
            "folder_map": {
                "architecture": "Projects/Vortex/Architecture",
                "session": "Projects/Vortex/Sessions",
                "decision": "Projects/Vortex/Decisions",
                "prompt": "Projects/Vortex/Prompts",
                "bug": "Projects/Vortex/Bugs",
                "experiment": "Projects/Vortex/Experiments",
            },
        },
    }


def test_spatial_state_store_open_update_and_event(tmp_path: Path) -> None:
    store = SpatialStateStore(
        settings=_settings(tmp_path),
        base_dir=tmp_path,
        panel_registry=PanelRegistry(),
    )

    opened = store.open_panel(
        {
            "type": "presentation",
            "title": "Deck",
            "content": "Slide 1",
            "source": {"pages": ["Slide 1", "Slide 2", "Slide 3"]},
            "region": {"x": 180, "y": 90, "width": 420, "height": 260},
        }
    )
    assert opened["ok"] is True
    panel = opened["panel"]
    assert panel["type"] == "presentation"
    assert panel["transform"]["x"] == 180.0
    assert panel["page_count"] == 3

    updated = store.update_panel(
        panel["id"],
        {
            "selected": True,
            "transform": {"rotation": 18, "tilt_y": 12},
        },
    )
    assert updated["ok"] is True
    assert updated["panel"]["transform"]["rotation"] == 18.0
    assert updated["panel"]["transform"]["tilt_y"] == 12.0

    event = store.apply_event(
        {
            "kind": "voice",
            "panel_id": panel["id"],
            "command": "open this presentation here",
            "region": {"x": 40, "y": 30, "width": 300, "height": 180},
        }
    )
    assert event["selected_object_id"] == panel["id"]
    assert event["last_voice_command"] == "open this presentation here"
    assert event["selected_region"]["width"] == 300.0
    assert "spatial panels=1" in event["recent_multimodal_summary"]


def test_obsidian_sync_service_saves_curated_note(tmp_path: Path) -> None:
    service = ObsidianSyncService(settings=_settings(tmp_path), base_dir=tmp_path)

    status = service.status()
    assert status["enabled"] is True
    assert status["available"] is False

    saved = service.save_note(
        note_type="decision",
        title="Spatial Fusion",
        content="Use sparse semantic fusion, not per-frame LLM calls.",
        metadata={"focus": "multimodal"},
        tags=["vortex", "spatial"],
    )
    assert saved["ok"] is True
    assert saved["path"].endswith(".md")

    note_path = Path(saved["path"])
    assert note_path.exists()
    body = note_path.read_text(encoding="utf-8")
    assert "type: decision" in body
    assert "focus: multimodal" in body
    assert "Use sparse semantic fusion" in body

    recent = service.iter_recent_notes(limit=4)
    assert len(recent) == 1
    assert "Spatial-Fusion" in recent[0]["title"]


def test_obsidian_sync_translates_windows_workspace_path_when_mounted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mount_root = tmp_path / "mounted-root"
    mount_root.mkdir(parents=True, exist_ok=True)
    settings = _settings(tmp_path)
    settings["obsidian"]["vault_path"] = r"D:\GitHub\Vortex\output\obsidian-vault"
    monkeypatch.setenv("C3RNT2_HOST_WORKSPACE_MOUNT", str(mount_root))
    monkeypatch.setenv("C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT", r"D:\GitHub\Vortex")
    monkeypatch.setenv("C3RNT2_HOST_WORKSPACE_REPO_NAME", "Vortex")

    service = ObsidianSyncService(settings=settings, base_dir=tmp_path)
    resolved = service.resolve_vault_path()
    assert resolved == (mount_root / "output" / "obsidian-vault").resolve()

    status = service.status()
    assert status["vault_path"] == r"D:\GitHub\Vortex\output\obsidian-vault"
    assert status["resolved_vault_path"] == str(resolved)


def test_extract_voice_intent_maps_spatial_commands() -> None:
    session = {"selected_object_id": "panel-1"}

    open_intent = extract_voice_intent("open this presentation here", session=session)
    tilt_intent = extract_voice_intent("tilt this panel", session=session)
    save_intent = extract_voice_intent("save this to obsidian", session=session)
    talk_intent = extract_voice_intent("talk to me about this", session=session)

    assert open_intent == {
        "kind": "open_panel",
        "panel_type": "presentation",
        "target": "selected_region",
        "title": "Spatial presentation",
        "panel_id": "panel-1",
    }
    assert tilt_intent == {
        "kind": "transform_panel",
        "panel_id": "panel-1",
        "transform": {"tilt_x": -10.0, "tilt_y": 14.0},
    }
    assert save_intent == {"kind": "save_obsidian", "panel_id": "panel-1"}
    assert talk_intent == {
        "kind": "chat_query",
        "panel_id": "panel-1",
        "query": "talk to me about this",
    }
