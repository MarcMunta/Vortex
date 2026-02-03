from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


RESTART_EXIT_CODE = 23
ROLLBACK_WINDOW_S = 3600.0
MAX_ROLLBACKS_PER_WINDOW = 2


def _log(event: str, **fields) -> None:
    payload = {"ts": time.time(), "event": event}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=True))


def _latest_backup_tag(cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "describe", "--tags", "--match", "autopilot/backup-*", "--abbrev=0"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    tag = (proc.stdout or "").strip()
    return tag or None


def _rollback_to_tag(cwd: Path, tag: str) -> bool:
    try:
        proc = subprocess.run(["git", "reset", "--hard", str(tag)], cwd=str(cwd))
    except Exception:
        return False
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_daemon.py")
    parser.add_argument("--profile", default="autonomous_4080_hf")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval-minutes", type=float, default=None)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backoff-s", type=float, default=2.0)
    parser.add_argument("--backoff-max-s", type=float, default=60.0)
    parser.add_argument("--cwd", default=None)
    args = parser.parse_args()

    base_dir = Path(args.cwd).resolve() if args.cwd else Path(__file__).resolve().parents[1]
    backoff = max(0.0, float(args.backoff_s))
    backoff_max = max(backoff, float(args.backoff_max_s))

    cmd = [
        sys.executable,
        "-m",
        "c3rnt2",
        "serve-autopilot",
        "--profile",
        str(args.profile),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
    ]
    if args.interval_minutes is not None:
        cmd += ["--interval-minutes", str(float(args.interval_minutes))]
    if args.no_web:
        cmd.append("--no-web")
    if args.force:
        cmd.append("--force")

    _log("daemon_start", cmd=cmd, cwd=str(base_dir))
    rollbacks: list[float] = []
    while True:
        start = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(base_dir))
        except KeyboardInterrupt:
            _log("daemon_stop", reason="keyboard_interrupt")
            return 0
        except Exception as exc:
            _log("daemon_error", error=str(exc))
            time.sleep(min(backoff_max, max(0.1, backoff)))
            backoff = min(backoff_max, max(backoff * 2.0, 0.5))
            continue

        elapsed = round(time.time() - start, 3)
        code = int(proc.returncode)
        if code == RESTART_EXIT_CODE:
            _log("daemon_restart", exit_code=code, elapsed_sec=elapsed)
            time.sleep(min(backoff_max, max(0.1, backoff)))
            backoff = min(backoff_max, max(backoff * 2.0, 0.5))
            continue

        if code == 0:
            _log("daemon_exit", exit_code=code, elapsed_sec=elapsed)
            return 0

        # Non-restart failure: attempt rollback to last known-good tag, then restart.
        now = time.time()
        rollbacks = [t for t in rollbacks if (now - t) <= ROLLBACK_WINDOW_S]
        if len(rollbacks) >= MAX_ROLLBACKS_PER_WINDOW:
            _log("daemon_exit_error", exit_code=code, elapsed_sec=elapsed, rollback="rate_limited")
            return code
        tag = _latest_backup_tag(base_dir)
        if not tag:
            _log("daemon_exit_error", exit_code=code, elapsed_sec=elapsed, rollback="no_backup_tag")
            return code
        if not _rollback_to_tag(base_dir, tag):
            _log("daemon_exit_error", exit_code=code, elapsed_sec=elapsed, rollback="reset_failed", tag=tag)
            return code
        rollbacks.append(now)
        _log("daemon_rollback", exit_code=code, elapsed_sec=elapsed, tag=tag)
        time.sleep(min(backoff_max, max(0.1, backoff)))
        backoff = min(backoff_max, max(backoff * 2.0, 0.5))
        continue


if __name__ == "__main__":
    raise SystemExit(main())
