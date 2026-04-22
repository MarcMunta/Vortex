from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if token]


def _embed_text(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        idx = hash(token) % dim
        vec[idx] += 1.0
    return vec


def _dot(left: list[float], right: list[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _safe_json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _normalize_timestamp(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = fallback
    return parsed if parsed > 0 else fallback


def _normalize_message(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    message_id = str(raw.get("id") or "").strip()
    role = str(raw.get("role") or "").strip().lower()
    content = str(raw.get("content") or "")
    if not message_id or role not in {"user", "ai"}:
        return None
    payload = dict(raw)
    payload["id"] = message_id
    payload["role"] = role
    payload["content"] = content
    payload["timestamp"] = _normalize_timestamp(payload.get("timestamp"), time.time() * 1000.0)
    if "thought" in payload and payload.get("thought") is not None:
        payload["thought"] = str(payload.get("thought") or "")
    if "requestId" in payload and payload.get("requestId") is not None:
        payload["requestId"] = str(payload.get("requestId") or "")
    return payload


def _normalize_session(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    session_id = str(raw.get("id") or "").strip()
    if not session_id:
        return None
    title = str(raw.get("title") or "Conversation").strip() or "Conversation"
    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, list):
        messages_raw = []
    messages = [item for item in (_normalize_message(msg) for msg in messages_raw) if item is not None]
    updated_at = _normalize_timestamp(raw.get("updatedAt"), time.time() * 1000.0)
    return {
        "id": session_id,
        "title": title,
        "messages": messages,
        "updatedAt": updated_at,
    }


def _message_text(title: str, message: dict[str, Any]) -> str:
    parts = [title.strip(), str(message.get("content") or "").strip()]
    thought = str(message.get("thought") or "").strip()
    if thought:
        parts.append(thought)
    return "\n".join(part for part in parts if part)


class ChatSessionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    account_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (account_id, session_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    account_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    title TEXT NOT NULL,
                    ts REAL NOT NULL,
                    text TEXT NOT NULL,
                    vec TEXT NOT NULL,
                    PRIMARY KEY (account_id, session_id, message_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_account_updated ON chat_sessions(account_id, updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_account_session_ts ON chat_messages(account_id, session_id, ts DESC)"
            )
            conn.commit()

    def list_sessions(self, account_id: str) -> list[dict[str, Any]]:
        account = str(account_id or "").strip()
        if not account:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM chat_sessions WHERE account_id = ? ORDER BY updated_at DESC, session_id DESC",
                (account,),
            ).fetchall()
        sessions: list[dict[str, Any]] = []
        for row in rows:
            payload = _safe_json_loads(str(row["payload"]), None)
            normalized = _normalize_session(payload)
            if normalized is not None:
                sessions.append(normalized)
        return sessions

    def clear_account(self, account_id: str) -> None:
        account = str(account_id or "").strip()
        if not account:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE account_id = ?", (account,))
            conn.execute("DELETE FROM chat_sessions WHERE account_id = ?", (account,))
            conn.commit()

    def delete_session(self, account_id: str, session_id: str) -> None:
        account = str(account_id or "").strip()
        session = str(session_id or "").strip()
        if not account or not session:
            return
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE account_id = ? AND session_id = ?",
                (account, session),
            )
            conn.execute(
                "DELETE FROM chat_sessions WHERE account_id = ? AND session_id = ?",
                (account, session),
            )
            conn.commit()

    def sync_sessions(
        self,
        account_id: str,
        sessions: list[Any],
        *,
        replace: bool = True,
    ) -> list[dict[str, Any]]:
        account = str(account_id or "").strip()
        if not account:
            return []
        normalized = [
            item for item in (_normalize_session(session) for session in sessions or []) if item is not None
        ]
        session_ids = [str(session["id"]) for session in normalized]
        with self._connect() as conn:
            if replace:
                if session_ids:
                    placeholders = ",".join("?" for _ in session_ids)
                    conn.execute(
                        f"DELETE FROM chat_messages WHERE account_id = ? AND session_id NOT IN ({placeholders})",
                        (account, *session_ids),
                    )
                    conn.execute(
                        f"DELETE FROM chat_sessions WHERE account_id = ? AND session_id NOT IN ({placeholders})",
                        (account, *session_ids),
                    )
                else:
                    conn.execute("DELETE FROM chat_messages WHERE account_id = ?", (account,))
                    conn.execute("DELETE FROM chat_sessions WHERE account_id = ?", (account,))
            for session in normalized:
                session_id = str(session["id"])
                payload_json = json.dumps(session, ensure_ascii=True, separators=(",", ":"))
                conn.execute(
                    """
                    INSERT INTO chat_sessions(account_id, session_id, title, updated_at, payload)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(account_id, session_id)
                    DO UPDATE SET title = excluded.title, updated_at = excluded.updated_at, payload = excluded.payload
                    """,
                    (
                        account,
                        session_id,
                        str(session.get("title") or "Conversation"),
                        float(session.get("updatedAt") or time.time() * 1000.0),
                        payload_json,
                    ),
                )
                conn.execute(
                    "DELETE FROM chat_messages WHERE account_id = ? AND session_id = ?",
                    (account, session_id),
                )
                title = str(session.get("title") or "Conversation")
                for message in session.get("messages", []):
                    if not isinstance(message, dict):
                        continue
                    text = _message_text(title, message).strip()
                    if not text:
                        continue
                    ts = _normalize_timestamp(message.get("timestamp"), time.time() * 1000.0)
                    vec = json.dumps(_embed_text(text), ensure_ascii=True, separators=(",", ":"))
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO chat_messages(
                            account_id, session_id, message_id, role, title, ts, text, vec
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            account,
                            session_id,
                            str(message.get("id") or ""),
                            str(message.get("role") or "user"),
                            title,
                            ts,
                            text[:8000],
                            vec,
                        ),
                    )
            conn.commit()
        return self.list_sessions(account)

    def query_relevant(
        self,
        account_id: str,
        query: str,
        *,
        session_id: str | None = None,
        exclude_message_ids: set[str] | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        account = str(account_id or "").strip()
        normalized_query = str(query or "").strip()
        if not account or not normalized_query:
            return []
        exclude = {str(item).strip() for item in (exclude_message_ids or set()) if str(item).strip()}
        qvec = _embed_text(normalized_query)
        qtokens = set(_tokenize(normalized_query))
        current_session = str(session_id or "").strip() or None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, message_id, role, title, ts, text, vec
                FROM chat_messages
                WHERE account_id = ?
                ORDER BY ts DESC
                """,
                (account,),
            ).fetchall()
        scored: list[dict[str, Any]] = []
        for row in rows:
            row_session = str(row["session_id"])
            row_message = str(row["message_id"])
            if row_session == current_session and row_message in exclude:
                continue
            text = str(row["text"] or "").strip()
            if not text:
                continue
            vec = _safe_json_loads(str(row["vec"]), [])
            if not isinstance(vec, list):
                vec = []
            score = _dot(qvec, [float(item) for item in vec if isinstance(item, (int, float))])
            tokens = set(_tokenize(text))
            overlap = len(qtokens.intersection(tokens))
            if overlap:
                score += float(overlap) * 0.35
            if row_session == current_session:
                score += 0.45
            if score <= 0.0:
                continue
            scored.append(
                {
                    "session_id": row_session,
                    "message_id": row_message,
                    "role": str(row["role"] or "user"),
                    "title": str(row["title"] or "Conversation"),
                    "ts": float(row["ts"] or 0.0),
                    "text": text,
                    "score": float(score),
                    "same_session": row_session == current_session,
                }
            )
        scored.sort(key=lambda item: (float(item["score"]), bool(item["same_session"]), float(item["ts"])), reverse=True)
        if scored:
            return scored[: max(1, int(top_k))]
        if not current_session:
            return []
        fallback: list[dict[str, Any]] = []
        for row in rows:
            if str(row["session_id"]) != current_session:
                continue
            row_message = str(row["message_id"])
            if row_message in exclude:
                continue
            text = str(row["text"] or "").strip()
            if not text:
                continue
            fallback.append(
                {
                    "session_id": current_session,
                    "message_id": row_message,
                    "role": str(row["role"] or "user"),
                    "title": str(row["title"] or "Conversation"),
                    "ts": float(row["ts"] or 0.0),
                    "text": text,
                    "score": 0.01,
                    "same_session": True,
                }
            )
            if len(fallback) >= max(1, int(top_k)):
                break
        return fallback

    def render_memory_block(
        self,
        account_id: str,
        query: str,
        *,
        session_id: str | None = None,
        exclude_message_ids: set[str] | None = None,
        top_k: int = 8,
        max_chars: int = 2400,
    ) -> tuple[str, list[dict[str, Any]]]:
        refs = self.query_relevant(
            account_id,
            query,
            session_id=session_id,
            exclude_message_ids=exclude_message_ids,
            top_k=top_k,
        )
        if not refs:
            return "", []
        lines: list[str] = []
        used = 0
        for ref in refs:
            scope = "same_session" if ref.get("same_session") else str(ref.get("title") or ref.get("session_id") or "history")
            role = "assistant" if str(ref.get("role") or "").lower() == "ai" else "user"
            text = str(ref.get("text") or "").replace("\r", " ").replace("\n", " ").strip()
            if len(text) > 420:
                text = text[:417].rstrip() + "..."
            line = f"- [{scope}] {role}: {text}"
            if used + len(line) > max_chars and lines:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return "", []
        header = (
            "CONVERSATION MEMORY (persistent history from this and previous chats; "
            "use only if it helps, and do not quote this block verbatim):"
        )
        return f"{header}\n" + "\n".join(lines), refs[: len(lines)]
