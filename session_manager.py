from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional


def _db_path() -> str:
    path = os.getenv("RESEARCH_DB_PATH", "/app/data/research.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
              session_id TEXT,
              role TEXT,
              content TEXT,
              created_at TIMESTAMP
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time ON chat_messages(session_id, created_at);"
        )
        conn.commit()
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Table: chat_messages
    Columns: session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP
    """
    sid = str(session_id or "").strip()
    if not sid:
        return
    db_path = _db_path()
    _init_db(db_path)
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_at) VALUES(?, ?, ?, CURRENT_TIMESTAMP)",
            (sid, str(role or "user"), str(content or "")),
        )
        conn.commit()
    finally:
        conn.close()


def load_recent_history(session_id: str, k: int = 5) -> list[dict]:
    """
    Return format:
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
    """
    sid = str(session_id or "").strip()
    if not sid:
        return []
    db_path = _db_path()
    _init_db(db_path)
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (sid, int(k)),
        ).fetchall()
        rows = list(reversed(rows))
        out: list[dict] = []
        for role, content in rows:
            out.append({"role": str(role or ""), "content": str(content or "")})
        return out
    finally:
        conn.close()


@dataclass(frozen=True)
class SessionMessage:
    role: str
    content: str
    created_at: int


class SessionManager:
    """
    Session history store (SQLite).
    Table: chat_messages(session_id, role, content, created_at)

    Demo-safe:
    - never raises
    - returns [] / False on failures
    """

    def __init__(self, *, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or _db_path()
        self._inited: bool = False

    def _init_if_needed(self) -> bool:
        if self._inited:
            return True
        try:
            conn = sqlite3.connect(self._db_path, timeout=5)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                      session_id TEXT,
                      role TEXT,
                      content TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time ON chat_messages(session_id, created_at);"
                )
                conn.commit()
                self._inited = True
                return True
            finally:
                conn.close()
        except Exception:
            return False

    # --- Required APIs ---
    def save_message(self, session_id: str, role: str, content: str) -> bool:
        try:
            sid = str(session_id or "").strip()
            if not sid:
                return False
            if not self._init_if_needed():
                return False
            conn = sqlite3.connect(self._db_path, timeout=5)
            try:
                conn.execute(
                    "INSERT INTO chat_messages(session_id, role, content, created_at) VALUES(?, ?, ?, CURRENT_TIMESTAMP)",
                    (sid, str(role or "user"), str(content or "")),
                )
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception:
            return False

    def load_recent_history(self, session_id: str, k: int = 5) -> List[Dict[str, str]]:
        """
        Return format:
        [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
        """
        try:
            sid = str(session_id or "").strip()
            if not sid:
                return []
            if not self._init_if_needed():
                return []
            conn = sqlite3.connect(self._db_path, timeout=5)
            try:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (sid, int(k)),
                ).fetchall()
                rows = list(reversed(rows))
                out: List[Dict[str, str]] = []
                for role, content, _created_at in rows:
                    out.append({"role": str(role or ""), "content": str(content or "")})
                return out
            finally:
                conn.close()
        except Exception:
            return []

    # --- Backward-compatible aliases (existing code) ---
    def append_message(self, session_id: str, role: str, content: str) -> bool:
        return self.save_message(session_id, role, content)

    def get_history(self, session_id: str, *, limit: int = 5) -> List[Dict[str, str]]:
        # Keep time field for older callers; still deterministic.
        try:
            sid = str(session_id or "").strip()
            if not sid:
                return []
            if not self._init_if_needed():
                return []
            conn = sqlite3.connect(self._db_path, timeout=5)
            try:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (sid, int(limit)),
                ).fetchall()
                rows = list(reversed(rows))
                out: List[Dict[str, str]] = []
                for role, content, created_at in rows:
                    out.append({"role": str(role or ""), "content": str(content or ""), "time": str(int(created_at or 0))})
                return out
            finally:
                conn.close()
        except Exception:
            return []

