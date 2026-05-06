from __future__ import annotations

from pathlib import Path

from c3rnt2.config import load_settings
from c3rnt2.context_budget import apply_message_budget, context_limit_for_mode, output_limit_for_mode
from c3rnt2.multimodal.obsidian_sync import ObsidianSyncService


def test_context_budget_defaults_are_safe_and_agent_is_larger() -> None:
    settings = load_settings("rtx4080_16gb_programming_qwen_coder_local")
    context = settings["context"]

    assert context["model_max_context_tokens"] == 32768
    assert context["default_chat_context_tokens"] == 2048
    assert context["default_agent_context_tokens"] == 4096
    assert context["max_output_tokens"] == 512
    assert context["max_agent_action_tokens"] == 512
    assert context["max_agent_final_tokens"] == 768
    assert context_limit_for_mode(settings, "agent") > context_limit_for_mode(settings, "chat")
    assert output_limit_for_mode(settings, "chat") == 512
    assert output_limit_for_mode(settings, "agent") == 512
    assert output_limit_for_mode(settings, "agent", final=True) == 768


def test_context_budget_summarizes_old_messages() -> None:
    settings = load_settings("rtx4080_16gb_programming_qwen_coder_local")
    messages = [{"role": "user", "content": f"old message {idx} " * 500} for idx in range(40)]

    budgeted = apply_message_budget(messages, settings, mode="chat")

    assert len(budgeted) < len(messages)
    assert any("Rolling conversation summary" in str(item.get("content")) for item in budgeted)
    assert "old message 39" in str(budgeted[-1]["content"])


def test_obsidian_missing_vault_does_not_fail(tmp_path: Path) -> None:
    settings = {
        "obsidian": {"enabled": True, "vault_path": str(tmp_path / "missing")},
        "context": {"obsidian_tokens": 1000},
    }
    service = ObsidianSyncService(settings=settings, base_dir=tmp_path)

    status = service.status()
    result = service.search("flutter navigation", top_k=3)

    assert status["ok"] is True
    assert status["available"] is False
    assert status["message"] == "Obsidian no configurado"
    assert result["ok"] is True
    assert result["notes"] == []


def test_obsidian_retrieves_markdown_notes_with_dedup_and_limits(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Flutter Navigation.md").write_text(
        "---\ntitle: Flutter Navigation\ntags: [flutter, dart]\n---\n"
        "# Navigator\nUse Navigator.push with MaterialPageRoute. [[Routes]]\n",
        encoding="utf-8",
    )
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "ignored.md").write_text("flutter secret cache", encoding="utf-8")
    service = ObsidianSyncService(
        settings={"obsidian": {"enabled": True, "vault_path": str(vault)}},
        base_dir=tmp_path,
    )

    indexed = service.reindex()
    found = service.search("Flutter Navigator route", top_k=5, max_tokens=100)
    context = service.build_context("Flutter Navigator route", top_k=5, max_tokens=100)

    assert indexed["notes"] == 1
    assert len(found["notes"]) == 1
    assert found["notes"][0]["relative_path"] == "Flutter Navigation.md"
    assert "Navigator.push" in found["notes"][0]["text"]
    assert "Curated Obsidian memory" in context["text"]


def test_cloud_training_placeholder_is_disabled_by_default() -> None:
    settings = load_settings("rtx4080_16gb_programming_qwen_coder_local")
    cloud_training = settings["cloud_training"]

    assert cloud_training["provider"] == "gcp"
    assert cloud_training["enabled"] is False
    assert cloud_training["service_account_env"] == "GOOGLE_APPLICATION_CREDENTIALS"
    assert "credential" not in str(cloud_training.get("project_id") or "").lower()
