from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional


class LockUnavailable(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: Optional[int] = None

    def acquire(self, blocking: bool = False, timeout_s: float | None = None, poll_interval_s: float = 0.1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        self._fd = os.open(self.path, flags)
        try:
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
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except OSError as exc:
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
        if self._fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire(blocking=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def acquire_exclusive_lock(base_dir: Path, role: str) -> FileLock:
    role = role.lower()
    roles = {"serve", "train", "self_patch"}
    if role not in roles:
        raise ValueError("role must be 'serve', 'train', or 'self_patch'")
    lock_dir = base_dir / "data" / "locks"
    own_path = lock_dir / f"{role}.lock"
    own_lock = FileLock(own_path)
    own_lock.acquire(blocking=False)
    for other_role in roles - {role}:
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
    role = role.lower()
    roles = {"serve", "train", "self_patch"}
    if role not in roles:
        raise ValueError("role must be 'serve', 'train', or 'self_patch'")
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
