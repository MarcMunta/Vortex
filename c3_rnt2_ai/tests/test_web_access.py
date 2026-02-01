import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from c3rnt2.tools.web_access import web_fetch, reset_rate_limits


def _run_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_web_fetch_allowlist_and_cache(tmp_path: Path):
    reset_rate_limits()
    hits = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits["count"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"hello world")

        def log_message(self, format, *args):
            return

    server = _run_server(Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    cache_dir = tmp_path / "cache"
    res = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=10, cache_ttl_s=3600)
    assert res.ok
    res2 = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=10, cache_ttl_s=3600)
    assert res2.ok
    assert hits["count"] == 1
    server.shutdown()


def test_web_fetch_rate_limit(tmp_path: Path):
    reset_rate_limits()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = _run_server(Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    cache_dir = tmp_path / "cache"
    res = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=1, cache_ttl_s=0)
    assert res.ok
    res2 = web_fetch(url, allowlist=["127.0.0.1"], cache_dir=cache_dir, rate_limit_per_min=1, cache_ttl_s=0)
    assert not res2.ok
    server.shutdown()


def test_web_fetch_allowlist_block(tmp_path: Path):
    reset_rate_limits()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = _run_server(Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    cache_dir = tmp_path / "cache"
    res = web_fetch(url, allowlist=["example.com"], cache_dir=cache_dir, rate_limit_per_min=10)
    assert not res.ok
    server.shutdown()
