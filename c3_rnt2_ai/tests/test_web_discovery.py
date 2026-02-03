from __future__ import annotations

from pathlib import Path

from c3rnt2.tools import web_discovery as wd
from c3rnt2.tools.web_access import WebFetchResult
from c3rnt2.tools.web_ingest import canonicalize_url


def test_web_discovery_filters_by_allow_domains_and_updates_state(tmp_path: Path, monkeypatch) -> None:
    html = (
        '<a href="/l/?kh=-1&uddg=https%3A%2F%2Fdocs.python.org%2F3%2Freference%2F">python</a>'
        '<a href="/l/?uddg=https%3A%2F%2Fevil.com%2Fprompt">evil</a>'
        '<a href="https://pytorch.org/docs/stable/">torch</a>'
    )

    def _fake_fetch(_url: str, _allowlist: list[str], **_kwargs):
        return WebFetchResult(ok=True, url=_url, status=200, text=html, from_cache=True, error=None)

    monkeypatch.setattr(wd, "web_fetch", _fake_fetch)

    state: dict = {}
    settings = {
        "tools": {"web": {"allow_domains": ["duckduckgo.com", "docs.python.org", "pytorch.org"], "cache_dir": str(tmp_path / "cache")}},
        "continuous": {"web_discovery": {"enabled": True, "seed_queries": ["python"], "max_urls_per_tick": 10}},
    }
    res = wd.discover_urls(settings, base_dir=tmp_path, state=state)
    assert res.get("ok") is True
    urls = state.get("discovered_urls", [])
    assert canonicalize_url("https://docs.python.org/3/reference/") in urls
    assert canonicalize_url("https://pytorch.org/docs/stable/") in urls
    assert all("evil.com" not in u for u in urls)

