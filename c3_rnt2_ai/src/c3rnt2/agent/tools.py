from __future__ import annotations

import html
import json
import re
import subprocess
from urllib.parse import quote_plus
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..tools.web_access import web_fetch
from ..utils.locks import is_lock_held
from .sandbox import run_sandbox_command


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict[str, object] = field(default_factory=dict)


class AgentTools:
    def __init__(
        self,
        allowlist: List[str],
        sandbox_root: Path | None = None,
        cache_root: Path | None = None,
        rate_limit_per_min: int = 30,
        web_cfg: dict | None = None,
        agent_cfg: dict | None = None,
        self_patch_cfg: dict | None = None,
        security_cfg: dict | None = None,
        repo_root: Path | None = None,
    ):
        raw_cfg: dict[str, Any] = dict(web_cfg or {})
        if "web" in raw_cfg:
            raw_cfg = raw_cfg.get("web", {}) or {}
        self.web_cfg: dict[str, Any] = raw_cfg
        self.agent_cfg: dict[str, Any] = dict(agent_cfg or {})
        self.self_patch_cfg: dict[str, Any] = dict(self_patch_cfg or {})
        self.security_cfg: dict[str, Any] = dict(security_cfg or {})
        security_web = self.security_cfg.get("web", {}) if isinstance(self.security_cfg.get("web", {}), dict) else {}
        self.web_strict = bool(security_web.get("strict", True))
        self.web_enabled = bool(self.web_cfg.get("enabled", False))
        security_allow = security_web.get("allowlist_domains") if isinstance(security_web, dict) else None
        if isinstance(security_allow, list):
            allow_domains = list(security_allow)
        else:
            allow_domains = self.web_cfg.get("allow_domains", allowlist) or []
        self.allowlist: list[str] = [str(item) for item in allow_domains if item]
        if self.web_enabled and not self.web_strict:
            normalized = [str(item).lower().strip() for item in self.allowlist if item]
            if "duckduckgo.com" not in normalized:
                self.allowlist.append("duckduckgo.com")
        self.cache_root: Path = Path(self.web_cfg.get("cache_dir") or cache_root or Path("data") / "web_cache")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_min: int = int(self.web_cfg.get("rate_limit_per_min", rate_limit_per_min))
        self.max_bytes: int = int(self.web_cfg.get("max_bytes", 512_000))
        self.timeout_s: int = int(self.web_cfg.get("timeout_s", 10))
        self.max_redirects: int = int(self.web_cfg.get("max_redirects", security_web.get("max_redirects", 5) if isinstance(security_web, dict) else 5))
        self.user_agent: str = str(self.web_cfg.get("user_agent", security_web.get("user_agent", "KlimeAI-WebFetch/1.0") if isinstance(security_web, dict) else "KlimeAI-WebFetch/1.0"))
        cache_ttl = self.web_cfg.get("cache_ttl_s", None)
        self.cache_ttl_s: int | None = int(cache_ttl) if cache_ttl is not None else None
        allow_content_types = self.web_cfg.get("allow_content_types")
        if isinstance(allow_content_types, list):
            self.allow_content_types: list[str] | None = [str(item) for item in allow_content_types if item]
        else:
            self.allow_content_types = None
        self.sandbox_root: Path = sandbox_root or Path("data") / "workspaces"
        self.allow_git = bool(self.agent_cfg.get("allow_git", False))
        self.repo_root: Path | None = repo_root.resolve() if repo_root else None

    def _allowed_bases(self) -> List[Path]:
        bases: List[Path] = []
        if self.repo_root:
            bases.append(self.repo_root)
        if self.sandbox_root:
            bases.append(self.sandbox_root.resolve())
        return bases

    def _is_allowed_path(self, path: Path) -> bool:
        for base in self._allowed_bases():
            if path == base or base in path.parents:
                return True
        return False

    def _resolve_safe_path(self, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            candidate = path.resolve()
            return candidate if self._is_allowed_path(candidate) else None
        for base in self._allowed_bases():
            candidate = (base / path).resolve()
            if self._is_allowed_path(candidate):
                return candidate
        return None

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
            timeout_s=int(self.timeout_s),
            cache_dir=self.cache_root,
            rate_limit_per_min=self.rate_limit_per_min,
            cache_ttl_s=self.cache_ttl_s,
            allow_content_types=self.allow_content_types,
            strict=bool(self.web_strict),
            max_redirects=int(self.max_redirects),
            user_agent=str(self.user_agent),
        )
        if not result.ok:
            return ToolResult(ok=False, output=result.error or "fetch failed", meta={"url": getattr(result, "url", url)})
        text = self._sanitize_text(result.text)
        meta = {
            "url": getattr(result, "url", url),
            "status": getattr(result, "status", None),
            "from_cache": getattr(result, "from_cache", False),
            "etag": getattr(result, "etag", None),
            "last_modified": getattr(result, "last_modified", None),
            "content_hash": getattr(result, "content_hash", None),
        }
        return ToolResult(ok=True, output=text, meta=meta)

    def fetch_page(self, url: str) -> ToolResult:
        return self.web_fetch(url)

    def search_web(self, query: str, max_results: int = 5) -> ToolResult:
        if not self.web_enabled:
            return ToolResult(ok=False, output="web disabled")
        if not query:
            return ToolResult(ok=False, output="query required")
        encoded = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded}"
        result = web_fetch(
            url,
            allowlist=self.allowlist,
            max_bytes=self.max_bytes,
            timeout_s=int(self.timeout_s),
            cache_dir=self.cache_root,
            rate_limit_per_min=self.rate_limit_per_min,
            cache_ttl_s=self.cache_ttl_s,
            allow_content_types=self.allow_content_types,
        )
        if not result.ok:
            return ToolResult(ok=False, output=result.error or "fetch failed")
        html_text = result.text or ""
        matches = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>',
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        lines: List[str] = []
        for href, title_html in matches:
            title = re.sub(r"<[^>]+>", " ", title_html)
            title = html.unescape(" ".join(title.split()))
            link = html.unescape(href)
            if not title or not link:
                continue
            lines.append(f"{title} - {link}")
            if len(lines) >= max_results:
                break
        if not lines:
            fallback = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.IGNORECASE | re.DOTALL)
            for href, title_html in fallback:
                title = re.sub(r"<[^>]+>", " ", title_html)
                title = html.unescape(" ".join(title.split()))
                link = html.unescape(href)
                if not title or not link:
                    continue
                lines.append(f"{title} - {link}")
                if len(lines) >= max_results:
                    break
        if not lines:
            cleaned = self._sanitize_text(html_text)
            return ToolResult(ok=True, output=cleaned[:800])
        return ToolResult(ok=True, output="\n".join(lines))

    def open_docs(self, url: str, max_chars: int | None = None) -> ToolResult:
        result = self.fetch_page(url)
        if not result.ok:
            return result
        limit = max_chars
        if limit is None:
            try:
                limit = int(self.web_cfg.get("open_docs_max_chars", 1200))
            except Exception:
                limit = 1200
        output = result.output
        if limit and len(output) > limit:
            output = output[:limit]
        return ToolResult(ok=True, output=output, meta=dict(result.meta))

    def run_tests(self, repo_path: Path) -> ToolResult:
        result = run_sandbox_command(repo_path, ["pytest", "-q"], self.sandbox_root, timeout_s=300)
        out = (result.get("stdout", "") + result.get("stderr", "")).strip()
        return ToolResult(ok=bool(result.get("ok")), output=out)

    def read_file(self, path: str, max_chars: int = 4000) -> ToolResult:
        try:
            target: Path | None = self._resolve_safe_path(path)
            if target is None or not target.exists() or not target.is_file():
                return ToolResult(ok=False, output="path not allowed")
            text = target.read_text(encoding="utf-8", errors="ignore")
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            return ToolResult(ok=True, output=text)
        except Exception as exc:
            return ToolResult(ok=False, output=f"read failed: {exc}")

    def list_tree(self, root: str = ".", max_entries: int = 200) -> ToolResult:
        try:
            base: Path | None = self._resolve_safe_path(root)
            if base is None or not base.exists() or not base.is_dir():
                return ToolResult(ok=False, output="path not allowed")
            entries: List[str] = []
            for path in sorted(base.rglob("*")):
                if len(entries) >= max_entries:
                    break
                try:
                    rel = path.relative_to(base)
                except Exception:
                    rel = Path(path.name)
                entries.append(rel.as_posix())
            return ToolResult(ok=True, output="\n".join(entries))
        except Exception as exc:
            return ToolResult(ok=False, output=f"list failed: {exc}")

    def grep(self, pattern: str, path_glob: str = "**/*", max_hits: int = 50) -> ToolResult:
        try:
            if not pattern:
                return ToolResult(ok=False, output="pattern required")
            regex = re.compile(pattern)
        except Exception as exc:
            return ToolResult(ok=False, output=f"invalid pattern: {exc}")
        try:
            if Path(path_glob).is_absolute():
                return ToolResult(ok=False, output="path_glob must be relative")
            base = self.repo_root or self.sandbox_root.resolve()
            if not self._is_allowed_path(base):
                return ToolResult(ok=False, output="base path not allowed")
            hits: List[str] = []
            for path in sorted(base.glob(path_glob)):
                if len(hits) >= max_hits:
                    break
                try:
                    resolved = path.resolve()
                except Exception:
                    continue
                if not resolved.is_file() or not self._is_allowed_path(resolved):
                    continue
                try:
                    text = resolved.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for idx, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        try:
                            rel = resolved.relative_to(base)
                        except Exception:
                            rel = resolved
                        hits.append(f"{rel.as_posix()}:{idx}:{line}")
                        if len(hits) >= max_hits:
                            break
            return ToolResult(ok=True, output="\n".join(hits))
        except Exception as exc:
            return ToolResult(ok=False, output=f"grep failed: {exc}")

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

    def propose_patch(
        self,
        repo_root: Path,
        changes: Dict[Path, str],
        goal: str = "agent_patch",
        *,
        llm_generate_diff: bool = False,
        llm_context: dict[str, object] | None = None,
    ) -> ToolResult:
        try:
            safety_cfg = self.self_patch_cfg.get("safety", {}) if isinstance(self.self_patch_cfg, dict) else {}
            if safety_cfg.get("forbid_self_patch_during_train") and self.repo_root and is_lock_held(self.repo_root, "train"):
                return ToolResult(ok=False, output="self_patch blocked during training")
            from ..self_patch.propose_patch import propose_patch

            context: dict[str, object] = {"changes": {str(k): v for k, v in changes.items()}}
            if llm_context:
                context.update(llm_context)
            files: List[str] = []
            files_value = context.get("files")
            if isinstance(files_value, list):
                files.extend([str(f) for f in files_value if f])
            tool_calls: List[dict[str, object]] = []
            if isinstance(llm_context, dict):
                calls_value = llm_context.get("tool_calls")
                if isinstance(calls_value, list):
                    tool_calls = [item for item in calls_value if isinstance(item, dict)]
            pytest_output = ""
            for call in reversed(tool_calls):
                if call.get("action") == "run_tests" and call.get("output"):
                    pytest_output = str(call.get("output", ""))
                    break
            if pytest_output:
                files.extend(self._extract_pytest_paths(pytest_output, repo_root))
            files = self._limit_context_files(files, repo_root)
            if files:
                context["files"] = files
            proposal = propose_patch(
                goal,
                context,
                repo_root,
                settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {},
                llm_generate_diff=llm_generate_diff,
            )
            return ToolResult(ok=True, output=proposal.patch_id)
        except Exception as exc:
            return ToolResult(ok=False, output=f"propose failed: {exc}")

    def _extract_pytest_paths(self, output: str, repo_root: Path) -> List[str]:
        if not output:
            return []
        repo_root = repo_root.resolve()
        pattern = re.compile(r"(?P<path>(?:[A-Za-z]:)?[\\w\\-./\\\\]+\\.py)")
        found: List[str] = []
        for match in pattern.finditer(output):
            raw = match.group("path")
            if not raw:
                continue
            path = Path(raw)
            if path.is_absolute():
                try:
                    rel = path.resolve().relative_to(repo_root)
                except Exception:
                    continue
                found.append(rel.as_posix())
            else:
                found.append(path.as_posix())
        # Prefer tests then src
        def _priority(p: str) -> tuple[int, str]:
            if "tests" in p.replace("\\", "/").split("/"):
                return (0, p)
            if "src" in p.replace("\\", "/").split("/"):
                return (1, p)
            return (2, p)
        deduped = []
        seen = set()
        for item in sorted(found, key=_priority):
            norm = item.replace("\\", "/")
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(norm)
        return deduped

    def _limit_context_files(self, files: List[str], repo_root: Path, *, max_chars: int = 16000, max_file_chars: int = 4000) -> List[str]:
        if not files:
            return []
        repo_root = repo_root.resolve()
        selected: List[str] = []
        used = 0
        for item in files:
            path = Path(item)
            if path.is_absolute():
                try:
                    path = path.resolve().relative_to(repo_root)
                except Exception:
                    continue
            full = (repo_root / path).resolve()
            if repo_root not in full.parents and full != repo_root:
                continue
            if not full.exists() or not full.is_file():
                continue
            try:
                size = full.stat().st_size
            except Exception:
                size = max_file_chars
            est = min(int(size), max_file_chars)
            if selected and used + est > max_chars:
                continue
            if not selected and est > max_chars:
                est = max_chars
            selected.append(path.as_posix())
            used += est
            if used >= max_chars:
                break
        return selected

    def sandbox_patch(self, repo_root: Path, patch_id: str) -> ToolResult:
        try:
            from ..self_patch.sandbox_run import sandbox_run

            result = sandbox_run(repo_root, patch_id, settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {})
            return ToolResult(ok=bool(result.get("ok")), output=json.dumps(result))
        except Exception as exc:
            return ToolResult(ok=False, output=f"sandbox failed: {exc}")

    def apply_patch(self, repo_root: Path, patch_id: str, approve: bool = False) -> ToolResult:
        try:
            if not approve:
                return ToolResult(ok=False, output="approval required")
            from ..self_patch.apply_patch import apply_patch

            result = apply_patch(patch_id, repo_root, settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {})
            return ToolResult(ok=result.ok, output=result.error or "applied")
        except Exception as exc:
            return ToolResult(ok=False, output=f"apply failed: {exc}")

    def write_patch(self, repo_root: Path, diff_text: str, goal: str = "agent_patch", create_branch: bool = False) -> ToolResult:
        if create_branch and not self.allow_git:
            return ToolResult(ok=False, output="git disabled")
        try:
            from ..self_patch.propose_patch import propose_patch

            proposal = propose_patch(goal, {"changes": {}}, repo_root, settings={"self_patch": self.self_patch_cfg} if self.self_patch_cfg else {}, diff_text=diff_text)
            branch = None
            if create_branch and self.allow_git:
                branch = f"agent/{proposal.patch_id}"
                subprocess.run(["git", "checkout", "-b", branch], cwd=str(repo_root), check=False)
            payload = {"patch_id": proposal.patch_id, "branch": branch}
            return ToolResult(ok=True, output=json.dumps(payload))
        except Exception as exc:
            return ToolResult(ok=False, output=f"write_patch failed: {exc}")

    def summarize_diff(self, repo_root: Path) -> ToolResult:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
            return ToolResult(ok=True, output=result.stdout.strip())
        except Exception as exc:
            return ToolResult(ok=False, output=f"diff failed: {exc}")
