from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BudgetDecision:
    ok: bool
    reason: str
    state_path: Path
    state: dict[str, Any]


def _state_path(base_dir: Path) -> Path:
    return base_dir / "data" / "continuous" / "budget_state.json"


def _date_key(ts: float | None = None) -> str:
    return time.strftime("%Y%m%d", time.localtime(ts or time.time()))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"date": _date_key(), "steps_used": 0, "tokens_used": 0, "ts": time.time()}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"date": _date_key(), "steps_used": 0, "tokens_used": 0, "ts": time.time()}
    if not isinstance(payload, dict):
        return {"date": _date_key(), "steps_used": 0, "tokens_used": 0, "ts": time.time()}
    return payload


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += int(p.stat().st_size)
        except Exception:
            continue
    return total


def can_start_run(
    base_dir: Path,
    *,
    max_steps_per_day: int | None,
    max_tokens_per_day: int | None,
    max_disk_mb: int | None,
    planned_steps: int | None = None,
    planned_tokens: int | None = None,
) -> BudgetDecision:
    state_path = _state_path(base_dir)
    state = _load_state(state_path)
    today = _date_key()
    if str(state.get("date")) != today:
        state = {"date": today, "steps_used": 0, "tokens_used": 0, "ts": time.time()}

    steps_used = int(state.get("steps_used", 0) or 0)
    tokens_used = int(state.get("tokens_used", 0) or 0)

    if max_steps_per_day is not None and int(max_steps_per_day) > 0:
        if steps_used >= int(max_steps_per_day):
            return BudgetDecision(ok=False, reason="max_steps_per_day_exceeded", state_path=state_path, state=state)
        if planned_steps is not None and steps_used + int(planned_steps) > int(max_steps_per_day):
            return BudgetDecision(ok=False, reason="max_steps_per_day_would_exceed", state_path=state_path, state=state)

    if max_tokens_per_day is not None and int(max_tokens_per_day) > 0:
        if tokens_used >= int(max_tokens_per_day):
            return BudgetDecision(ok=False, reason="max_tokens_per_day_exceeded", state_path=state_path, state=state)
        if planned_tokens is not None and tokens_used + int(planned_tokens) > int(max_tokens_per_day):
            return BudgetDecision(ok=False, reason="max_tokens_per_day_would_exceed", state_path=state_path, state=state)

    if max_disk_mb is not None and int(max_disk_mb) > 0:
        cont_root = base_dir / "data" / "continuous"
        used_mb = _dir_size_bytes(cont_root) / 1e6
        if used_mb >= float(max_disk_mb):
            return BudgetDecision(
                ok=False,
                reason=f"max_disk_mb_exceeded:{used_mb:.1f}",
                state_path=state_path,
                state=state,
            )

    return BudgetDecision(ok=True, reason="ok", state_path=state_path, state=state)


def record_run(base_dir: Path, *, steps: int, tokens: int) -> BudgetDecision:
    state_path = _state_path(base_dir)
    state = _load_state(state_path)
    today = _date_key()
    if str(state.get("date")) != today:
        state = {"date": today, "steps_used": 0, "tokens_used": 0, "ts": time.time()}
    state["steps_used"] = int(state.get("steps_used", 0) or 0) + int(steps)
    state["tokens_used"] = int(state.get("tokens_used", 0) or 0) + int(tokens)
    state["ts"] = time.time()
    _atomic_write(state_path, state)
    return BudgetDecision(ok=True, reason="recorded", state_path=state_path, state=state)

