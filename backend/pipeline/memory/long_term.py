"""Long-term memory stored in SQLite — persists across server restarts."""
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict
import threading


class LongTermMemory:
    def __init__(self, db_path: str = "data/memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    title        TEXT DEFAULT 'New Chat',
                    created_at   TEXT NOT NULL,
                    last_active  TEXT NOT NULL,
                    summary      TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    timestamp    TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
            """)
            conn.commit()

    def upsert_session(self, session_id: str, title: str = "New Chat"):
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, title, created_at, last_active) VALUES (?,?,?,?)",
                    (session_id, title, now, now),
                )
                conn.execute(
                    "UPDATE sessions SET last_active=? WHERE session_id=?",
                    (now, session_id),
                )
                conn.commit()

    def save_message(self, session_id: str, role: str, content: str):
        self.upsert_session(session_id)
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    (session_id, role, content, now),
                )
                conn.execute(
                    "UPDATE sessions SET last_active=? WHERE session_id=?",
                    (now, session_id),
                )
                conn.commit()

    def save_turn(self, session_id: str, user_content: str, assistant_content: str):
        """Persist both sides of a turn in one transaction."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, title, created_at, last_active) VALUES (?,?,?,?)",
                    (session_id, "New Chat", now, now),
                )
                conn.executemany(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    [
                        (session_id, "user", user_content, now),
                        (session_id, "assistant", assistant_content, now),
                    ],
                )
                conn.execute(
                    "UPDATE sessions SET last_active=? WHERE session_id=?",
                    (now, session_id),
                )

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cur.fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    def list_sessions(self) -> List[Dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT session_id, title, created_at, last_active, summary "
                "FROM sessions ORDER BY last_active DESC"
            )
            rows = cur.fetchall()
        return [
            {"session_id": r[0], "title": r[1], "created_at": r[2], "last_active": r[3], "summary": r[4]}
            for r in rows
        ]

    def update_session_title(self, session_id: str, title: str):
        with self._lock:
            with self._conn() as conn:
                conn.execute("UPDATE sessions SET title=? WHERE session_id=?", (title, session_id))
                conn.commit()

    def update_summary(self, session_id: str, summary: str):
        with self._lock:
            with self._conn() as conn:
                conn.execute("UPDATE sessions SET summary=? WHERE session_id=?", (summary, session_id))
                conn.commit()

    def delete_session(self, session_id: str):
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
                conn.commit()
