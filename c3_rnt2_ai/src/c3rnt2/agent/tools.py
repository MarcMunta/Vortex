from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..tools.web_access import web_fetch


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
        raw_cfg = dict(web_cfg or {})
        if "web" in raw_cfg:
            raw_cfg = raw_cfg.get("web", {}) or {}
        self.web_cfg = raw_cfg
        self.web_enabled = bool(self.web_cfg.get("enabled", False))
        self.allowlist = list(self.web_cfg.get("allow_domains", allowlist) or [])
        self.cache_root = Path(self.web_cfg.get("cache_dir") or cache_root or Path("data") / "web_cache")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_min = int(self.web_cfg.get("rate_limit_per_min", rate_limit_per_min))
        self.max_bytes = int(self.web_cfg.get("max_bytes", 512_000))
        self.timeout_s = float(self.web_cfg.get("timeout_s", 10))
        cache_ttl = self.web_cfg.get("cache_ttl_s", None)
        self.cache_ttl_s = int(cache_ttl) if cache_ttl is not None else None
        self.allow_content_types = self.web_cfg.get("allow_content_types")
        self.sandbox_root = sandbox_root or Path("data") / "workspaces"

    def _sanitize_text(self, text: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def web_fetch(self, url: str) -> ToolResult:
        if not self.web_enabled:
            return ToolResult(ok=False, output="web disabled")
        result = web_fetch(
            url,
            allowlist=self.allowlist,
            max_bytes=self.max_bytes,
            timeout_s=self.timeout_s,
            cache_dir=self.cache_root,
            rate_limit_per_min=self.rate_limit_per_min,
            cache_ttl_s=self.cache_ttl_s,
            allow_content_types=self.allow_content_types,
        )
        if not result.ok:
            return ToolResult(ok=False, output=result.error or "fetch failed")
        text = self._sanitize_text(result.text)
        return ToolResult(ok=True, output=text)

    def search_web(self, query: str) -> ToolResult:
        url = f"https://duckduckgo.com/html/?q={query}"
        result = self.web_fetch(url)
        if not result.ok:
            return result
        return ToolResult(ok=True, output=result.output[:1000])

    def open_docs(self, url: str) -> ToolResult:
        result = self.web_fetch(url)
        if not result.ok:
            return result
        return ToolResult(ok=True, output=result.output[:1200])

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

    def propose_patch(self, repo_root: Path, changes: Dict[Path, str], goal: str = "agent_patch") -> ToolResult:
        try:
            from ..self_patch.propose_patch import propose_patch

            context = {"changes": {str(k): v for k, v in changes.items()}}
            proposal = propose_patch(goal, context, repo_root, settings={})
            return ToolResult(ok=True, output=proposal.patch_id)
        except Exception as exc:
            return ToolResult(ok=False, output=f"propose failed: {exc}")

    def sandbox_patch(self, repo_root: Path, patch_id: str) -> ToolResult:
        try:
            from ..self_patch.sandbox_run import sandbox_run

            result = sandbox_run(repo_root, patch_id, settings={})
            return ToolResult(ok=bool(result.get("ok")), output=json.dumps(result))
        except Exception as exc:
            return ToolResult(ok=False, output=f"sandbox failed: {exc}")

    def apply_patch(self, repo_root: Path, patch_id: str, approve: bool = False) -> ToolResult:
        try:
            if not approve:
                return ToolResult(ok=False, output="approval required")
            from ..self_patch.apply_patch import apply_patch

            result = apply_patch(patch_id, repo_root, settings={})
            return ToolResult(ok=result.ok, output=result.error or "applied")
        except Exception as exc:
            return ToolResult(ok=False, output=f"apply failed: {exc}")
