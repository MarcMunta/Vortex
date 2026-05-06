from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported_type:{type(value)!r}")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_default)


def _loads(raw: str | bytes | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


class OperationalStore:
    """Versioned SQLite store for control-plane operational state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state_records (
                  key TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  schema_version INTEGER NOT NULL,
                  updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_records (
                  scope TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  event_id TEXT NOT NULL,
                  ts REAL NOT NULL,
                  payload_json TEXT NOT NULL,
                  schema_version INTEGER NOT NULL,
                  PRIMARY KEY (scope, entity_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_event_records_scope_ts
                  ON event_records(scope, ts DESC);
                CREATE TABLE IF NOT EXISTS run_records (
                  run_id TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  schema_version INTEGER NOT NULL,
                  created_at REAL,
                  updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_records_updated
                  ON run_records(updated_at DESC);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def get_state(self, key: str, default: Any) -> Any:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM state_records WHERE key = ?",
                (key,),
            ).fetchone()
        return _loads(row["payload_json"], default) if row else default

    def put_state(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        updated_at = float(payload.get("updated_at") or time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state_records(key, payload_json, schema_version, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  schema_version=excluded.schema_version,
                  updated_at=excluded.updated_at
                """,
                (key, _dumps(payload), SCHEMA_VERSION, updated_at),
            )
        return payload

    def append_event(self, scope: str, entity_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = str(payload.get("id") or f"{scope}-{entity_id}-{time.time_ns()}")
        ts = float(payload.get("ts") or time.time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO event_records(scope, entity_id, event_id, ts, payload_json, schema_version)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (scope, entity_id, event_id, ts, _dumps(payload), SCHEMA_VERSION),
            )
        return payload

    def list_events(
        self,
        scope: str,
        entity_id: str | None = None,
        *,
        limit: int | None = None,
        reverse: bool = True,
    ) -> list[dict[str, Any]]:
        direction = "DESC" if reverse else "ASC"
        params: list[Any] = [scope]
        query = "SELECT payload_json FROM event_records WHERE scope = ?"
        if entity_id is not None:
            query += " AND entity_id = ?"
            params.append(entity_id)
        query += f" ORDER BY ts {direction}"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [item for item in (_loads(row["payload_json"], {}) for row in rows) if isinstance(item, dict)]

    def put_run(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        created_at = float(payload.get("created_at") or payload.get("updated_at") or time.time())
        updated_at = float(payload.get("updated_at") or created_at)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_records(run_id, payload_json, schema_version, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  schema_version=excluded.schema_version,
                  created_at=COALESCE(run_records.created_at, excluded.created_at),
                  updated_at=excluded.updated_at
                """,
                (run_id, _dumps(payload), SCHEMA_VERSION, created_at, updated_at),
            )
        return payload

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM run_records WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        payload = _loads(row["payload_json"], None) if row else None
        return payload if isinstance(payload, dict) else None

    def list_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        params: list[Any] = []
        query = "SELECT payload_json FROM run_records ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [item for item in (_loads(row["payload_json"], {}) for row in rows) if isinstance(item, dict)]

    def delete_all_runs(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM run_records")
            conn.execute("DELETE FROM event_records WHERE scope = 'training_run'")

    def clear_events(self, scope: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM event_records WHERE scope = ?", (scope,))

    def import_state_file(self, key: str, path: Path, default: Any) -> None:
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = default
        if isinstance(payload, dict):
            self.put_state(key, payload)

    def import_jsonl_events(self, scope: str, entity_id: str, path: Path) -> None:
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            payload = _loads(raw, {})
            if isinstance(payload, dict):
                self.append_event(scope, str(payload.get("run_id") or entity_id), payload)

    def import_training_runs(self, runs_dir: Path) -> None:
        if not runs_dir.exists():
            return
        for meta_path in runs_dir.glob("*/meta.json"):
            payload = _loads(meta_path.read_text(encoding="utf-8", errors="ignore"), {})
            if not isinstance(payload, dict):
                continue
            run_id = str(payload.get("run_id") or meta_path.parent.name).strip()
            if not run_id:
                continue
            payload.setdefault("run_id", run_id)
            self.put_run(run_id, payload)
            self.import_jsonl_events("training_run", run_id, meta_path.parent / "events.jsonl")
