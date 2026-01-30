from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .patch_ops import PatchResult, apply_patch, propose_patch, validate_patch
from .safety_kernel import SafetyPolicy


def _extract_failures(output: str) -> list[dict]:
    tasks = []
    for line in output.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            tasks.append({"issue": line.strip()})
    if not tasks:
        # fallback: include tail of output
        tail = "\n".join(output.splitlines()[-5:])
        if tail:
            tasks.append({"issue": tail})
    return tasks

def run_improve_loop(repo_root: Path) -> dict:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "data" / "selfimprove" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    policy = SafetyPolicy()
    report = {"run_id": run_id, "status": "started"}

    try:
        result = subprocess.run(["pytest", "-q"], cwd=str(repo_root), capture_output=True, text=True)
        if result.returncode == 0:
            report["status"] = "tests_ok"
            report["tests_output"] = result.stdout[:500]
        else:
            report["status"] = "tests_failed"
            output = (result.stdout + result.stderr)
            report["tests_output"] = output[:500]
            repair_tasks = _extract_failures(output)
            (run_dir / "repair_tasks.json").write_text(json.dumps(repair_tasks), encoding="utf-8")
            diff = propose_patch(repo_root, {})
            (run_dir / "proposed.diff").write_text(diff, encoding="utf-8")
            validation = validate_patch(repo_root, diff, policy)
            report["validation"] = {"ok": validation.ok, "message": validation.message}
        (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report
