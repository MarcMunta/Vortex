from __future__ import annotations

import subprocess
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


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
        web_cfg: dict | None = None,
    ):
        self.policy = WebPolicy(allowlist=allowlist, rate_limit_per_min=rate_limit_per_min)
        self.web_cfg = web_cfg or {}
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


    def web_fetch(self, url: str) -> ToolResult:
        cfg = self.web_cfg.get("web", {}) if isinstance(self.web_cfg, dict) else {}
        if not cfg.get("enabled", False):
            return ToolResult(ok=False, output="web disabled")
        from ..tools.web_access import web_fetch
        allow_domains = cfg.get("allow_domains", self.policy.allowlist)
        max_bytes = int(cfg.get("max_bytes", 512000))
        timeout_s = int(cfg.get("timeout_s", 10))
        cache_dir = Path(cfg.get("cache_dir", self.cache_root))
        rate_limit = int(cfg.get("rate_limit_per_min", self.policy.rate_limit_per_min))
        cache_ttl_s = cfg.get("cache_ttl_s", 3600)
        if cache_ttl_s is not None:
            cache_ttl_s = int(cache_ttl_s)
        allow_content_types = cfg.get("allow_content_types")
        result = web_fetch(
            url,
            allowlist=list(allow_domains),
            max_bytes=max_bytes,
            timeout_s=timeout_s,
            cache_dir=cache_dir,
            rate_limit_per_min=rate_limit,
            cache_ttl_s=cache_ttl_s,
            allow_content_types=allow_content_types,
            base_dir=Path('.'),
        )
        if not result.ok:
            return ToolResult(ok=False, output=result.error or "fetch failed")
        text = self._sanitize_text(result.text)
        return ToolResult(ok=True, output=text)

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
        url = f"https://duckduckgo.com/html/?q={query}"
        return self.web_fetch(url)

    def open_docs(self, url: str) -> ToolResult:
        return self.web_fetch(url)

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
