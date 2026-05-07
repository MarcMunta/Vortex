from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Optional


class LockUnavailable(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: str | Path):
        # Usamos sqlite3 como manejador robusto de locks.
        # Cambiamos la extensión a .db para reflejar su nueva naturaleza.
        original_path = Path(path)
        self.path = original_path.with_suffix('.db')
        self._conn: Optional[sqlite3.Connection] = None
        self._held: bool = False

    def acquire(self, blocking: bool = False, timeout_s: float | None = None, poll_interval_s: float = 0.1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        deadline = None
        if blocking:
            try:
                if timeout_s is not None:
                    timeout_val = float(timeout_s)
                    if timeout_val <= 0:
                        timeout_val = 0.0
                    deadline = time.monotonic() + timeout_val
            except Exception:
                deadline = None

        while True:
            try:
                self._conn = sqlite3.connect(str(self.path), timeout=0.01, isolation_level="EXCLUSIVE")
                # Crear tabla dummy si no existe
                self._conn.execute("CREATE TABLE IF NOT EXISTS _lock (id INTEGER PRIMARY KEY)")
                self._conn.execute("PRAGMA locking_mode = EXCLUSIVE")
                self._conn.execute("BEGIN EXCLUSIVE")
                self._held = True
                return
            except sqlite3.OperationalError as exc:
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    
                if not blocking:
                    raise LockUnavailable(str(exc))
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockUnavailable("timeout")
                try:
                    sleep_s = max(0.01, float(poll_interval_s))
                except Exception:
                    sleep_s = 0.1
                time.sleep(sleep_s)
            except Exception:
                self.release()
                raise

    def release(self) -> None:
        if self._conn is None:
            return
        try:
            if self._held:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
        finally:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire(blocking=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def acquire_exclusive_lock(base_dir: Path, role: str) -> FileLock:
    base_dir = Path(base_dir)
    role = role.lower()
    roles = {"serve", "serve_fallback", "train", "self_patch"}
    if role not in roles:
        raise ValueError("role must be 'serve', 'serve_fallback', 'train', or 'self_patch'")
    conflicts = {
        "serve": {"serve_fallback", "train", "self_patch"},
        "serve_fallback": {"serve", "self_patch"},
        "train": {"serve", "self_patch"},
        "self_patch": {"serve", "serve_fallback", "train"},
    }
    lock_dir = base_dir / "data" / "locks"
    own_path = lock_dir / f"{role}.lock"
    own_lock = FileLock(own_path)
    own_lock.acquire(blocking=False)
    for other_role in conflicts.get(role, set()):
        other_path = lock_dir / f"{other_role}.lock"
        other_lock = FileLock(other_path)
        try:
            other_lock.acquire(blocking=False)
            other_lock.release()
        except LockUnavailable:
            own_lock.release()
            raise
    return own_lock


def is_lock_held(base_dir: Path, role: str) -> bool:
    base_dir = Path(base_dir)
    role = role.lower()
    roles = {"serve", "serve_fallback", "train", "self_patch"}
    if role not in roles:
        raise ValueError("role must be 'serve', 'serve_fallback', 'train', or 'self_patch'")
    lock_dir = base_dir / "data" / "locks"
    path = lock_dir / f"{role}.lock"
    lock = FileLock(path)
    try:
        lock.acquire(blocking=False)
    except LockUnavailable:
        return True
    finally:
        lock.release()
    return False
