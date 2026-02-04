from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WslStatus:
    ok: bool
    error: str | None = None
    stdout: str | None = None
    stderr: str | None = None


def is_wsl_available(*, timeout_s: float = 1.5) -> WslStatus:
    try:
        res = subprocess.run(
            ["wsl.exe", "--status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s) if timeout_s is not None else 1.5,
        )
    except FileNotFoundError:
        return WslStatus(ok=False, error="wsl.exe not found")
    except subprocess.TimeoutExpired:
        return WslStatus(ok=False, error="wsl.exe --status timeout")
    except Exception as exc:  # pragma: no cover
        return WslStatus(ok=False, error=str(exc))

    if res.returncode != 0:
        msg = (res.stderr or "").strip() or (res.stdout or "").strip() or f"wsl_status_failed:{res.returncode}"
        return WslStatus(ok=False, error=msg, stdout=res.stdout, stderr=res.stderr)
    return WslStatus(ok=True, stdout=res.stdout, stderr=res.stderr)


def build_bash_lc_script(
    cmd: list[str],
    *,
    workdir: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    prefix = ""
    if env:
        parts = []
        for key, val in env.items():
            if not key:
                continue
            parts.append(f"{key}={shlex.quote(str(val))}")
        if parts:
            prefix = " ".join(parts) + " "
    cmd_str = " ".join(shlex.quote(str(part)) for part in cmd if part is not None)
    steps = []
    if workdir:
        steps.append(f"cd {shlex.quote(str(workdir))}")
    steps.append(f"{prefix}{cmd_str}".strip())
    return " && ".join(steps)


def build_wsl_bash_command(script: str) -> list[str]:
    # Use a deterministic invocation to keep output parsing reliable.
    return ["wsl.exe", "-e", "bash", "-lc", str(script)]


def parse_last_json_object(stdout: str) -> dict[str, Any] | None:
    import json

    payload = None
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    return payload if isinstance(payload, dict) else None

