from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)


@dataclass
class EpisodeRef:
    request_id: str
    path: Path
    offset: int
    ts: float


class EpisodeIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._prefer_wal = True
        self._sqlite_available = False
        self._memory_index: dict[str, EpisodeRef] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            if self._prefer_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            else:
                conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _disable_sqlite(self, exc: Exception) -> None:
        if self._sqlite_available:
            LOG.warning("episode index sqlite degraded to memory fallback: %s", exc)
        else:
            LOG.warning("episode index sqlite unavailable, using memory fallback: %s", exc)
        self._sqlite_available = False

    def _init_db(self) -> None:
        for prefer_wal in (True, False):
            self._prefer_wal = prefer_wal
            try:
                with self._connect() as conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS episode_index (request_id TEXT PRIMARY KEY, path TEXT, offset INTEGER, ts REAL)"
                    )
                    conn.commit()
                self._sqlite_available = True
                return
            except sqlite3.Error as exc:
                last_exc = exc
                continue
        self._disable_sqlite(last_exc)

    def add(self, request_id: str, path: Path, offset: int, ts: float) -> None:
        if not request_id:
            return
        ref = EpisodeRef(request_id=request_id, path=path, offset=int(offset), ts=float(ts))
        self._memory_index[request_id] = ref
        if not self._sqlite_available:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO episode_index (request_id, path, offset, ts) VALUES (?, ?, ?, ?)",
                    (request_id, str(path), int(offset), float(ts)),
                )
                conn.commit()
        except sqlite3.Error as exc:
            self._disable_sqlite(exc)

    def get(self, request_id: str) -> EpisodeRef | None:
        if not request_id:
            return None
        fallback_ref = self._memory_index.get(request_id)
        if not self._sqlite_available:
            return fallback_ref
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT request_id, path, offset, ts FROM episode_index WHERE request_id = ?",
                    (request_id,),
                )
                row = cur.fetchone()
        except sqlite3.Error as exc:
            self._disable_sqlite(exc)
            return fallback_ref
        if not row:
            return fallback_ref
        return EpisodeRef(request_id=str(row[0]), path=Path(str(row[1])), offset=int(row[2]), ts=float(row[3]))

    def load(self, request_id: str) -> dict | None:
        ref = self.get(request_id)
        if ref is None or not ref.path.exists():
            return None
        try:
            with ref.path.open("rb") as handle:
                handle.seek(ref.offset)
                line = handle.readline()
        except Exception:
            return None
        if not line:
            return None
        try:
            payload = json.loads(line.decode("utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None
