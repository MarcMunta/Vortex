from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_run_daemon_module() -> object:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_daemon.py"
    spec = importlib.util.spec_from_file_location("run_daemon", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_run_daemon_rolls_back_and_restarts(tmp_path: Path, monkeypatch) -> None:
    mod = _load_run_daemon_module()

    calls: list[list[str]] = []

    def _fake_run(cmd, cwd=None, capture_output=False, text=False, check=False):  # type: ignore[no-untyped-def]
        _ = check
        calls.append([str(x) for x in cmd])
        if cmd[:3] == ["git", "describe", "--tags"]:
            return SimpleNamespace(returncode=0, stdout="autopilot/backup-20260203\n")
        if cmd[:3] == ["git", "reset", "--hard"]:
            return SimpleNamespace(returncode=0)
        # serve-autopilot: fail once, then succeed.
        serve_calls = [c for c in calls if "-m" in c and "c3rnt2" in c and "serve-autopilot" in c]
        if len(serve_calls) == 1:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run=_fake_run))
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mod, "MAX_ROLLBACKS_PER_WINDOW", 5)

    # Run with explicit cwd so git commands are anchored.
    monkeypatch.setattr(mod.sys, "argv", ["run_daemon.py", "--cwd", str(tmp_path), "--backoff-s", "0.0"])
    code = mod.main()
    assert code == 0
    assert any(c[:3] == ["git", "reset", "--hard"] for c in calls)
