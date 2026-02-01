from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class LockUnavailable(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: Optional[int] = None

    def acquire(self, blocking: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        self._fd = os.open(self.path, flags)
        try:
            if os.name == "nt":
                import msvcrt

                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                try:
                    msvcrt.locking(self._fd, mode, 1)
                except OSError as exc:
                    raise LockUnavailable(str(exc))
            else:
                import fcntl

                lock_flags = fcntl.LOCK_EX
                if not blocking:
                    lock_flags |= fcntl.LOCK_NB
                try:
                    fcntl.flock(self._fd, lock_flags)
                except OSError as exc:
                    raise LockUnavailable(str(exc))
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
    other_roles = roles - {role}
    try:
        for other_role in other_roles:
            other_lock = FileLock(lock_dir / f"{other_role}.lock")
            other_lock.acquire(blocking=False)
            other_lock.release()
    except LockUnavailable:
        own_lock.release()
        raise
    return own_lock
