from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from c3rnt2.tools.web_access import reset_rate_limits, web_fetch


class _Handler(BaseHTTPRequestHandler):
    cache_hits = 0
    rate_hits = 0

    def log_message(self, format, *args):  # noqa: N802
        return

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/cache"):
            _Handler.cache_hits += 1
            body = b"cached-response"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/rate"):
            _Handler.rate_hits += 1
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_web_allowlist_blocks(tmp_path, monkeypatch):
    reset_rate_limits()
    monkeypatch.delenv("C3RNT2_NO_NET", raising=False)
    server = _start_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/cache"
        result = web_fetch(
            url,
            allowlist=["example.com"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=5,
            strict=False,
        )
        assert not result.ok
        assert result.error == "allowlist_blocked"
    finally:
        server.shutdown()
        server.server_close()


def test_web_strict_blocks_ip_literals(tmp_path, monkeypatch):
    reset_rate_limits()
    monkeypatch.delenv("C3RNT2_NO_NET", raising=False)
    result = web_fetch(
        "http://127.0.0.1:1234/test",
        allowlist=["127.0.0.1"],
        max_bytes=64,
        timeout_s=1,
        cache_dir=tmp_path / "cache",
        rate_limit_per_min=5,
        strict=True,
    )
    assert not result.ok
    assert result.error == "ip_literal_blocked"


def test_web_strict_blocks_file_scheme(tmp_path, monkeypatch):
    reset_rate_limits()
    monkeypatch.delenv("C3RNT2_NO_NET", raising=False)
    result = web_fetch(
        "file:///etc/passwd",
        allowlist=["example.com"],
        max_bytes=64,
        timeout_s=1,
        cache_dir=tmp_path / "cache",
        rate_limit_per_min=5,
        strict=True,
    )
    assert not result.ok
    assert str(result.error).startswith("scheme_blocked:")


def test_web_cache_hits(tmp_path, monkeypatch):
    reset_rate_limits()
    monkeypatch.delenv("C3RNT2_NO_NET", raising=False)
    _Handler.cache_hits = 0
    server = _start_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/cache"
        first = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=10,
            cache_ttl_s=3600,
            strict=False,
        )
        second = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=10,
            cache_ttl_s=3600,
            strict=False,
        )
        assert first.ok
        assert second.ok
        assert second.from_cache
        assert _Handler.cache_hits == 1
    finally:
        server.shutdown()
        server.server_close()


def test_web_rate_limit(tmp_path, monkeypatch):
    reset_rate_limits()
    monkeypatch.delenv("C3RNT2_NO_NET", raising=False)
    _Handler.rate_hits = 0
    server = _start_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/rate"
        first = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=1,
            cache_ttl_s=0,
            strict=False,
        )
        second = web_fetch(
            url,
            allowlist=["127.0.0.1"],
            max_bytes=1024,
            timeout_s=2,
            cache_dir=tmp_path / "cache",
            rate_limit_per_min=1,
            cache_ttl_s=0,
            strict=False,
        )
        assert first.ok
        assert not second.ok
        assert second.error == "rate_limited"
        assert _Handler.rate_hits == 1
    finally:
        server.shutdown()
        server.server_close()
