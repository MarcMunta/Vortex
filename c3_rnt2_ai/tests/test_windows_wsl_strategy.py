from __future__ import annotations

import subprocess
from pathlib import Path

from c3rnt2 import __main__ as main_mod
from c3rnt2.utils import wsl as wsl_mod


def test_wsl_subprocess_strategy_builds_wsl_exe_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_mod.sys, "platform", "win32", raising=False)

    monkeypatch.setattr(wsl_mod, "is_wsl_available", lambda timeout_s=1.5: wsl_mod.WslStatus(ok=True))

    seen: dict[str, object] = {}

    def _fake_run(cmd, check=False, capture_output=False, text=False, timeout=None, env=None):  # type: ignore[no-untyped-def]
        _ = check, capture_output, text, timeout, env
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok": true, "ok_train": true, "ok_eval": true}\n', stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", _fake_run)

    settings = {
        "_profile": "rtx4080_16gb_120b_like",
        "server": {"wsl_python": "python", "wsl_workdir": "/mnt/c/repo/c3_rnt2_ai"},
    }
    payload = main_mod._run_train_subprocess_wsl(settings, reuse_dataset=False, env={"C3RNT2_TRAIN_MAX_STEPS": "1"})
    assert payload.get("ok") is True

    cmd = seen.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[0].lower() == "wsl.exe"
    assert "-lc" in cmd
    script = str(cmd[-1])
    assert "python" in script
    assert " -m " in script
    assert "c3rnt2" in script
    assert "train-once" in script

