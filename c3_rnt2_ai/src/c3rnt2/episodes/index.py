from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


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
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS episode_index (request_id TEXT PRIMARY KEY, path TEXT, offset INTEGER, ts REAL)"
            )
            conn.commit()

    def add(self, request_id: str, path: Path, offset: int, ts: float) -> None:
        if not request_id:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episode_index (request_id, path, offset, ts) VALUES (?, ?, ?, ?)",
                (request_id, str(path), int(offset), float(ts)),
            )
            conn.commit()

    def get(self, request_id: str) -> EpisodeRef | None:
        if not request_id:
            return None
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT request_id, path, offset, ts FROM episode_index WHERE request_id = ?",
                (request_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
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
