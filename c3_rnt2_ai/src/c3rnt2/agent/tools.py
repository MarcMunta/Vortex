from __future__ import annotations

import subprocess
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .policies import WebPolicy


@dataclass
class ToolResult:
    ok: bool
    output: str


class AgentTools:
    def __init__(
        self,
        allowlist: List[str],
        sandbox_root: Path | None = None,
        cache_root: Path | None = None,
        rate_limit_per_min: int = 30,
    ):
        self.policy = WebPolicy(allowlist=allowlist, rate_limit_per_min=rate_limit_per_min)
        self.sandbox_root = sandbox_root or Path("data") / "workspaces"
        self.cache_root = cache_root or Path("data") / "web_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _sanitize_text(self, text: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_root / f"{digest}.json"

    def _cache_get(self, key: str) -> Optional[str]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("text")
        except Exception:
            return None

    def _cache_set(self, key: str, text: str) -> None:
        path = self._cache_path(key)
        payload = {"text": text}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def run_tests(self, repo_path: Path) -> ToolResult:
        try:
            result = subprocess.run(
                ["pytest", "-q"],
                cwd=str(repo_path),
                check=False,
                capture_output=True,
                text=True,
            )
            out = result.stdout + result.stderr
            return ToolResult(ok=result.returncode == 0, output=out.strip())
        except Exception as exc:
            return ToolResult(ok=False, output=f"pytest failed: {exc}")

    def search_web(self, query: str) -> ToolResult:
        if not self.policy.check_rate():
            return ToolResult(ok=False, output="rate limit exceeded")
        # MVP: naive GET to duckduckgo html if allowed (unlikely). Return stub if blocked.
        url = f"https://duckduckgo.com/html/?q={query}"
        if not self.policy.allow_url(url):
            return ToolResult(ok=False, output="domain not in allowlist")
        cached = self._cache_get(url)
        if cached is not None:
            return ToolResult(ok=True, output=cached)
        try:
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return ToolResult(ok=False, output=f"http {resp.status_code}")
            text = self._sanitize_text(resp.text)[:1000]
            self._cache_set(url, text)
            return ToolResult(ok=True, output=text)
        except Exception as exc:
            return ToolResult(ok=False, output=f"web error: {exc}")

    def open_docs(self, url: str) -> ToolResult:
        if not self.policy.check_rate():
            return ToolResult(ok=False, output="rate limit exceeded")
        if not self.policy.allow_url(url):
            return ToolResult(ok=False, output="domain not in allowlist")
        cached = self._cache_get(url)
        if cached is not None:
            return ToolResult(ok=True, output=cached)
        try:
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                return ToolResult(ok=False, output=f"http {resp.status_code}")
            text = self._sanitize_text(resp.text)[:1200]
            self._cache_set(url, text)
            return ToolResult(ok=True, output=text)
        except Exception as exc:
            return ToolResult(ok=False, output=f"web error: {exc}")

    def edit_repo(self, file_path: Path, new_text: str) -> ToolResult:
        try:
            self.sandbox_root.mkdir(parents=True, exist_ok=True)
            # Ensure edits only happen in sandbox workspace
            if not str(file_path).startswith(str(self.sandbox_root)):
                file_path = self.sandbox_root / file_path.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_text, encoding="utf-8")
            return ToolResult(ok=True, output=str(file_path))
        except Exception as exc:
            return ToolResult(ok=False, output=f"edit failed: {exc}")

    def propose_patch(self, repo_root: Path, changes: Dict[Path, str]) -> ToolResult:
        try:
            from ..selfimprove.patch_ops import propose_patch

            diff = propose_patch(repo_root, changes)
            return ToolResult(ok=True, output=diff)
        except Exception as exc:
            return ToolResult(ok=False, output=f"propose failed: {exc}")

    def validate_patch(self, repo_root: Path, diff_text: str) -> ToolResult:
        try:
            from ..selfimprove.patch_ops import validate_patch
            from ..selfimprove.safety_kernel import SafetyPolicy

            result = validate_patch(repo_root, diff_text, SafetyPolicy())
            return ToolResult(ok=result.ok, output=result.message)
        except Exception as exc:
            return ToolResult(ok=False, output=f"validate failed: {exc}")

    def apply_patch(self, repo_root: Path, diff_text: str, approve: bool = False) -> ToolResult:
        try:
            from ..selfimprove.patch_ops import apply_patch

            result = apply_patch(repo_root, diff_text, approve=approve)
            return ToolResult(ok=result.ok, output=result.message)
        except Exception as exc:
            return ToolResult(ok=False, output=f"apply failed: {exc}")
