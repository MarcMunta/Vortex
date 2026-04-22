from __future__ import annotations

from c3rnt2.config import load_settings


def test_multimodal_blocks_present_in_programming_profiles() -> None:
    local = load_settings("rtx4080_16gb_programming_local")
    runtime = load_settings("rtx4080_16gb_programming_runtime_docker")

    for settings in (local, runtime):
        assert settings["voice"]["enabled"] is True
        assert settings["voice"]["push_to_talk"] is True
        assert settings["camera"]["enabled"] is True
        assert settings["gesture"]["enabled"] is True
        assert settings["spatial_ui"]["enabled"] is True
        assert settings["spatial_ui"]["default_perspective"] == 1100
        assert settings["obsidian"]["enabled"] is True
        assert settings["multimodal_memory"]["enabled"] is True
        assert settings["multimodal_context"]["enabled"] is True
        assert settings["presentation"]["swipe_enabled"] is True
        assert settings["workspace_panels"]["max_active_panels"] == 12
        assert "presentation" in settings["workspace_panels"]["default_kinds"]


def test_multimodal_obsidian_folder_map_normalized() -> None:
    settings = load_settings("rtx4080_16gb_programming_local")
    folder_map = settings["obsidian"]["folder_map"]

    assert folder_map["architecture"] == "Projects/Vortex/Architecture"
    assert folder_map["session"] == "Projects/Vortex/Sessions"
    assert folder_map["decision"] == "Projects/Vortex/Decisions"
    assert folder_map["prompt"] == "Projects/Vortex/Prompts"
    assert folder_map["bug"] == "Projects/Vortex/Bugs"
    assert folder_map["experiment"] == "Projects/Vortex/Experiments"
