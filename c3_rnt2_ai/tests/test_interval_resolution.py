from __future__ import annotations

from c3rnt2.__main__ import _resolve_interval_minutes


def test_interval_minutes_fallback() -> None:
    settings = {"continuous": {"interval_minutes": 12}}
    assert _resolve_interval_minutes(None, settings) == 12.0
