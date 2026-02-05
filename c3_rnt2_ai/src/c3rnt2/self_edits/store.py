from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from c3rnt2.self_patch.utils import diff_paths, generate_diff
from c3rnt2.utils.locks import FileLock, LockUnavailable


class SelfEditsError(RuntimeError):
    pass


def find_repo_root(start_dir: Path) -> Path:
    start_dir = Path(start_dir).resolve()
    for candidate in [start_dir] + list(start_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return start_dir


def _now_ts() -> float:
    return float(time.time())


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _normalize_rel(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _is_forbidden(rel: str, forbidden_globs: Iterable[str]) -> bool:
    rel = rel.replace("\\", "/")
    for pattern in forbidden_globs:
        pat = str(pattern).replace("\\", "/")
        if pat.endswith("/") and rel.startswith(pat):
            return True
        if fnmatch.fnmatch(rel, pat):
            return True
    return False


def _is_allowed(rel: str, allowed_paths: Iterable[str], forbidden_globs: Iterable[str]) -> bool:
    rel = rel.replace("\\", "/")
    if _is_forbidden(rel, forbidden_globs):
        return False
    for allow in allowed_paths:
        allow_norm = str(allow).replace("\\", "/")
        if allow_norm.endswith("/"):
            if rel.startswith(allow_norm):
                return True
            continue
        if rel == allow_norm:
            return True
    return False


def _touched_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    for p in diff_paths(diff_text):
        rel = _normalize_rel(p)
        if rel and rel not in paths:
            paths.append(rel)
    return paths


def _extract_file_changes(diff_text: str, *, max_files: int = 50, max_chars_per_file: int = 32_000) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    current_path: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_path, current_lines, files
        if not current_path:
            current_lines = []
            return
        text = "\n".join(current_lines).strip("\n")
        if max_chars_per_file > 0 and len(text) > max_chars_per_file:
            text = text[: max(0, max_chars_per_file - 20)] + "\n... (truncated)\n"
        files.append({"path": current_path, "diff": text})
        current_path = None
        current_lines = []

    for raw in diff_text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("diff --git "):
            continue
        if line.startswith("--- a/"):
            if current_path is not None:
                _flush()
            continue
        if line.startswith("+++ b/"):
            current_path = line[6:].strip()
            continue
        if current_path is None:
            continue
        if line.startswith("index ") or line.startswith("new file mode") or line.startswith("deleted file mode"):
            continue
        if line.startswith("rename from ") or line.startswith("rename to ") or line.startswith("similarity index "):
            continue
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            continue
        current_lines.append(line)
        if len(files) >= max_files:
            break
    if current_path is not None and len(files) < max_files:
        _flush()
    return files


def _build_offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env["C3RNT2_NO_NET"] = "1"
    env["NO_PROXY"] = "*"
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["http_proxy"] = ""
    env["https_proxy"] = ""
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


def _run_cmd(cmd: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str]) -> dict[str, Any]:
    started = _now_ts()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env, timeout=float(timeout_s))
        ended = _now_ts()
        return {
            "cmd": " ".join(cmd),
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "duration_s": round(ended - started, 3),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        ended = _now_ts()
        return {
            "cmd": " ".join(cmd),
            "returncode": -1,
            "stdout_tail": (getattr(exc, "stdout", "") or "")[-4000:],
            "stderr_tail": (getattr(exc, "stderr", "") or "")[-4000:],
            "duration_s": round(ended - started, 3),
            "timeout": True,
        }
    except Exception as exc:
        ended = _now_ts()
        return {
            "cmd": " ".join(cmd),
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "duration_s": round(ended - started, 3),
            "timeout": False,
        }


@dataclass
class SelfEditsStore:
    app_dir: Path
    repo_root: Path
    profile: str

    proposals_dir: Path
    sandbox_dir: Path
    lock_path: Path

    max_patch_kb: int = 256
    max_list: int = 200

    allowed_paths: list[str] | None = None
    forbidden_globs: list[str] | None = None

    @staticmethod
    def from_app_dir(app_dir: Path, *, profile: str | None = None) -> "SelfEditsStore":
        app_dir = Path(app_dir).resolve()
        repo_root = find_repo_root(app_dir)
        resolved_profile = str(profile or os.getenv("C3RNT2_PROFILE") or "dev_small").strip() or "dev_small"

        proposals_dir = app_dir / "skills" / "_proposals" / "self_edits"
        sandbox_dir = app_dir / "data" / "self_edits" / "sandbox"
        lock_path = app_dir / "data" / "locks" / "self_edits.lock"

        allowed_paths = [
            "README.md",
            "dev.bat",
            "dev_cli.ps1",
            "run.bat",
            "stop.bat",
            "status.bat",
            "logs.bat",
            "vortex-chat/",
            "c3_rnt2_ai/src/",
            "c3_rnt2_ai/tests/",
            "c3_rnt2_ai/scripts/",
            "c3_rnt2_ai/config/",
        ]
        forbidden_globs = [
            ".env",
            ".env.*",
            ".git/**",
            "data/**",
            "c3_rnt2_ai/data/**",
            "vortex-chat/node_modules/**",
            "vortex-chat/dist/**",
            "vortex-chat/.vite/**",
            "c3_rnt2_ai/skills/_staging/**",
            "c3_rnt2_ai/skills/_proposals/**",
            "*.key",
            "*.pem",
            "*.p12",
            "*.sqlite",
            "*.db",
            "keys/**",
            "secrets/**",
        ]
        return SelfEditsStore(
            app_dir=app_dir,
            repo_root=repo_root,
            profile=resolved_profile,
            proposals_dir=proposals_dir,
            sandbox_dir=sandbox_dir,
            lock_path=lock_path,
            allowed_paths=allowed_paths,
            forbidden_globs=forbidden_globs,
        )

    def _proposal_dir(self, proposal_id: str) -> Path:
        pid = str(proposal_id or "").strip()
        if not pid or "/" in pid or "\\" in pid or ".." in pid:
            raise SelfEditsError("invalid_proposal_id")
        return self.proposals_dir / pid

    def list_proposals(self, *, status: str | None = None) -> list[dict[str, Any]]:
        root = self.proposals_dir
        if not root.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
            if len(items) >= int(self.max_list):
                break
            if not path.is_dir():
                continue
            meta_path = path / "meta.json"
            meta = _safe_read_json(meta_path) if meta_path.exists() else {}
            pid = str(meta.get("id") or meta.get("patch_id") or path.name)
            st = str(meta.get("status") or "pending")
            api_status = "pending" if st in {"proposed", "ready_for_review"} else st
            if status:
                if status == "pending":
                    if st not in {"pending", "accepted", "proposed", "ready_for_review"}:
                        continue
                elif api_status != status:
                    continue
            title = str(meta.get("title") or meta.get("goal") or f"Proposal {pid}")
            summary = str(meta.get("summary") or meta.get("goal") or "")
            created = meta.get("created_at") or meta.get("created_ts") or meta.get("created") or None
            try:
                created_ts = float(created) if created is not None else None
            except Exception:
                created_ts = None
            payload: dict[str, Any] = {
                "id": pid,
                "created_at": created_ts,
                "title": title,
                "summary": summary,
                "status": api_status,
                "author": str(meta.get("author") or "agent"),
            }
            try:
                patch_path = path / "patch.diff"
                if patch_path.exists():
                    diff_text = patch_path.read_text(encoding="utf-8", errors="ignore")
                    payload["files"] = [p for p in _touched_paths(diff_text)]
                else:
                    payload["files"] = []
            except Exception:
                payload["files"] = []
            items.append(payload)
        return items

    def get(self, proposal_id: str) -> dict[str, Any]:
        pdir = self._proposal_dir(proposal_id)
        if not pdir.exists():
            raise SelfEditsError("proposal_not_found")
        meta_path = pdir / "meta.json"
        patch_path = pdir / "patch.diff"
        meta = _safe_read_json(meta_path) if meta_path.exists() else {}
        diff_text = patch_path.read_text(encoding="utf-8", errors="ignore") if patch_path.exists() else ""
        pid = str(meta.get("id") or meta.get("patch_id") or pdir.name)
        st = str(meta.get("status") or "pending")
        normalized = st
        if st in {"proposed", "ready_for_review"}:
            normalized = "pending"
        created = meta.get("created_at") or meta.get("created_ts") or meta.get("created") or None
        try:
            created_ts = float(created) if created is not None else None
        except Exception:
            created_ts = None
        payload: dict[str, Any] = {
            "id": pid,
            "created_at": created_ts,
            "title": str(meta.get("title") or meta.get("goal") or f"Proposal {pid}"),
            "summary": str(meta.get("summary") or meta.get("goal") or ""),
            "status": normalized,
            "author": str(meta.get("author") or "agent"),
            "files": [p for p in _touched_paths(diff_text)],
            "diff": diff_text,
            "fileChanges": _extract_file_changes(diff_text),
            "sandbox": _safe_read_json(pdir / "sandbox.json") if (pdir / "sandbox.json").exists() else None,
            "apply": _safe_read_json(pdir / "apply.json") if (pdir / "apply.json").exists() else None,
        }
        return payload

    def accept(self, proposal_id: str) -> dict[str, Any]:
        pdir = self._proposal_dir(proposal_id)
        meta_path = pdir / "meta.json"
        if not meta_path.exists():
            raise SelfEditsError("meta.json not found")
        meta = _safe_read_json(meta_path)
        st = str(meta.get("status") or "pending")
        if st in {"rejected", "applied"}:
            raise SelfEditsError(f"cannot_accept_status:{st}")
        meta["status"] = "accepted"
        meta["accepted_at"] = _now_ts()
        meta.setdefault("id", pdir.name)
        _atomic_write_json(meta_path, meta)
        return {"ok": True, "id": str(meta.get("id") or pdir.name), "status": "accepted"}

    def reject(self, proposal_id: str) -> dict[str, Any]:
        pdir = self._proposal_dir(proposal_id)
        meta_path = pdir / "meta.json"
        if not meta_path.exists():
            raise SelfEditsError("meta.json not found")
        meta = _safe_read_json(meta_path)
        st = str(meta.get("status") or "pending")
        if st in {"applied"}:
            raise SelfEditsError(f"cannot_reject_status:{st}")
        meta["status"] = "rejected"
        meta["rejected_at"] = _now_ts()
        meta.setdefault("id", pdir.name)
        _atomic_write_json(meta_path, meta)
        return {"ok": True, "id": str(meta.get("id") or pdir.name), "status": "rejected"}

    def create_demo(self) -> dict[str, Any]:
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        pid = f"demo-{int(time.time())}"
        pdir = self._proposal_dir(pid)
        if pdir.exists():
            raise SelfEditsError("demo_already_exists")
        pdir.mkdir(parents=True, exist_ok=True)
        patch_path = pdir / "patch.diff"
        demo_file = Path("c3_rnt2_ai/src/c3rnt2/_self_edits_demo.txt")
        diff_text = generate_diff(
            self.repo_root,
            {
                demo_file: "Self-edits demo proposal.\nSafe pipeline: propose -> accept -> apply (tests + rollback).\n",
            },
        )
        patch_path.write_text(diff_text, encoding="utf-8")
        meta = {
            "id": pid,
            "created_at": _now_ts(),
            "title": "Demo: Self-Edits proposal",
            "summary": "Adds a tiny demo text file under c3_rnt2_ai/src/ (safe).",
            "author": "agent",
            "status": "pending",
        }
        _atomic_write_json(pdir / "meta.json", meta)
        return {"ok": True, "id": pid}

    def _acquire_lock(self) -> FileLock:
        lock = FileLock(self.lock_path)
        lock.acquire(blocking=False)
        return lock

    def _validate_patch(self, diff_text: str) -> tuple[bool, str | None, list[str]]:
        max_kb = int(self.max_patch_kb)
        if max_kb > 0 and len(diff_text.encode("utf-8")) > max_kb * 1024:
            return False, "patch_exceeds_max_patch_kb", []
        touched = _touched_paths(diff_text)
        if not touched:
            return False, "patch_has_no_files", []
        allowed = list(self.allowed_paths or [])
        forbidden = list(self.forbidden_globs or [])
        for rel in touched:
            if not _is_allowed(rel, allowed, forbidden):
                return False, f"forbidden_path:{rel}", touched
        return True, None, touched

    def _ensure_git_clean(self) -> tuple[bool, str | None]:
        try:
            proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(self.repo_root), capture_output=True, text=True)
        except Exception as exc:
            return False, f"git_status_failed:{exc}"
        if proc.returncode != 0:
            return False, (proc.stderr or "").strip() or "git_status_failed"
        if (proc.stdout or "").strip():
            return False, "working_tree_dirty"
        return True, None

    def _ensure_tracked_or_new(self, rel_paths: list[str]) -> tuple[bool, str | None, dict[str, bool]]:
        existed_before: dict[str, bool] = {}
        for rel in rel_paths:
            abs_path = (self.repo_root / rel).resolve()
            existed_before[rel] = abs_path.exists()
            if abs_path.exists():
                try:
                    proc = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=str(self.repo_root), capture_output=True, text=True)
                except Exception as exc:
                    return False, f"git_ls_files_failed:{exc}", existed_before
                if proc.returncode != 0:
                    return False, f"untracked_existing_file:{rel}", existed_before
        return True, None, existed_before

    def _rollback_paths(self, rel_paths: list[str], existed_before: dict[str, bool]) -> None:
        for rel in rel_paths:
            if existed_before.get(rel, False):
                subprocess.run(["git", "checkout", "--", rel], cwd=str(self.repo_root), check=False, capture_output=True, text=True)
            else:
                try:
                    abs_path = (self.repo_root / rel).resolve()
                    if abs_path.exists():
                        if abs_path.is_dir():
                            continue
                        abs_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def apply(self, proposal_id: str) -> dict[str, Any]:
        pdir = self._proposal_dir(proposal_id)
        meta_path = pdir / "meta.json"
        patch_path = pdir / "patch.diff"
        if not meta_path.exists() or not patch_path.exists():
            raise SelfEditsError("proposal_incomplete")

        meta = _safe_read_json(meta_path)
        status = str(meta.get("status") or "pending")
        if status != "accepted":
            raise SelfEditsError("apply_requires_accepted")

        diff_text = patch_path.read_text(encoding="utf-8", errors="ignore")
        ok_patch, patch_err, touched = self._validate_patch(diff_text)
        if not ok_patch:
            meta["status"] = "failed"
            meta["error"] = patch_err
            meta["failed_at"] = _now_ts()
            _atomic_write_json(meta_path, meta)
            return {"ok": False, "error": patch_err, "id": str(meta.get("id") or pdir.name)}

        lock = None
        try:
            lock = self._acquire_lock()
        except LockUnavailable:
            return {"ok": False, "error": "self_edits_busy"}
        try:
            ok_clean, clean_err = self._ensure_git_clean()
            if not ok_clean:
                return {"ok": False, "error": clean_err or "working_tree_dirty"}

            ok_tracked, tracked_err, existed_before = self._ensure_tracked_or_new(touched)
            if not ok_tracked:
                return {"ok": False, "error": tracked_err or "untracked_file"}

            env = _build_offline_env()

            apply_out = _run_cmd(["git", "apply", str(patch_path)], cwd=self.repo_root, timeout_s=30.0, env=env)
            if apply_out["returncode"] != 0:
                meta["status"] = "failed"
                meta["error"] = "apply_failed"
                meta["failed_at"] = _now_ts()
                _atomic_write_json(meta_path, meta)
                _atomic_write_json(pdir / "apply.json", {"ok": False, "stage": "git_apply", "apply": apply_out})
                return {"ok": False, "error": "apply_failed", "details": apply_out}

            checks: list[dict[str, Any]] = []
            checks.append(_run_cmd(["pytest", "-q"], cwd=self.repo_root, timeout_s=900.0, env=env))
            checks.append(_run_cmd([sys.executable, "-m", "c3rnt2", "skills", "validate", "--all"], cwd=self.app_dir, timeout_s=120.0, env=env))
            checks.append(_run_cmd([sys.executable, "-m", "c3rnt2", "doctor", "--deep", "--mock", "--profile", self.profile], cwd=self.app_dir, timeout_s=600.0, env=env))

            touched_frontend = any(p.startswith("vortex-chat/") for p in touched)
            if touched_frontend and (self.repo_root / "vortex-chat" / "package.json").exists():
                checks.append(_run_cmd(["npm", "run", "build"], cwd=self.repo_root / "vortex-chat", timeout_s=900.0, env=env))

            ok_all = True
            for item in checks:
                rc = int(item.get("returncode", 1))
                cmd = str(item.get("cmd") or "")
                if cmd.startswith("pytest") and rc in (0, 5):
                    continue
                if rc != 0:
                    ok_all = False
                    break

            if not ok_all:
                subprocess.run(["git", "apply", "-R", str(patch_path)], cwd=str(self.repo_root), check=False, capture_output=True, text=True)
                self._rollback_paths(touched, existed_before)
                meta["status"] = "failed"
                meta["error"] = "validation_failed_rolled_back"
                meta["failed_at"] = _now_ts()
                _atomic_write_json(meta_path, meta)
                _atomic_write_json(pdir / "apply.json", {"ok": False, "stage": "validate", "checks": checks, "rolled_back": True})
                return {"ok": False, "error": "validation_failed_rolled_back", "checks": checks}

            meta["status"] = "applied"
            meta["applied_at"] = _now_ts()
            _atomic_write_json(meta_path, meta)
            _atomic_write_json(pdir / "apply.json", {"ok": True, "checks": checks, "applied_at": meta["applied_at"]})
            return {"ok": True, "id": str(meta.get("id") or pdir.name), "status": "applied", "checks": checks}
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass
