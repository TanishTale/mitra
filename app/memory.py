"""Persistence: episodic (turn-by-turn), semantic (rolling summary) and
structured (mood, journal) memory, all in a local SQLite file.

Privacy stance: data never leaves the machine, the user can wipe a session with
one call, and consent is recorded per session. No analytics, no telemetry.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

def _now() -> str:
    """UTC timestamp, timezone-aware."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    consent     INTEGER NOT NULL DEFAULT 1,
    summary     TEXT DEFAULT '',
    turns       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    risk_level  TEXT DEFAULT '',
    risk_score  REAL DEFAULT 0,
    intent      TEXT DEFAULT '',
    ts          TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS moods (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    score       INTEGER NOT NULL,
    label       TEXT DEFAULT '',
    note        TEXT DEFAULT '',
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    prompt      TEXT DEFAULT '',
    entry       TEXT NOT NULL,
    tags        TEXT DEFAULT '[]',
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    risk_score  REAL NOT NULL,
    signals     TEXT NOT NULL,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_moods_session ON moods(session_id);
"""


class MemoryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- sessions -------------------------------------------------------
    def new_session(self, consent: bool = True) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions (id, created_at, consent) VALUES (?,?,?)",
                (sid, _now(), int(consent)),
            )
        return sid

    def ensure_session(self, sid: Optional[str], consent: bool = True) -> str:
        if not sid:
            return self.new_session(consent)
        with self._conn() as c:
            row = c.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
        return sid if row else self.new_session(consent)

    def session_info(self, sid: str) -> Dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else {}

    def set_summary(self, sid: str, summary: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET summary=? WHERE id=?", (summary, sid))

    def bump_turns(self, sid: str) -> int:
        with self._conn() as c:
            c.execute("UPDATE sessions SET turns = turns + 1 WHERE id=?", (sid,))
            row = c.execute("SELECT turns FROM sessions WHERE id=?", (sid,)).fetchone()
        return row["turns"] if row else 0

    # ---- messages -------------------------------------------------------
    def add_message(self, sid: str, role: str, content: str, *,
                    risk_level: str = "", risk_score: float = 0.0,
                    intent: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages (session_id, role, content, risk_level, risk_score,"
                " intent, ts) VALUES (?,?,?,?,?,?,?)",
                (sid, role, content, risk_level, risk_score, intent,
                 _now()),
            )

    def history(self, sid: str, limit: int = 12) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, ts FROM messages WHERE session_id=?"
                " ORDER BY id DESC LIMIT ?", (sid, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def user_turns(self, sid: str, limit: int = 6) -> List[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT content FROM messages WHERE session_id=? AND role='user'"
                " ORDER BY id DESC LIMIT ?", (sid, limit),
            ).fetchall()
        return [r["content"] for r in reversed(rows)]

    # ---- moods ----------------------------------------------------------
    def log_mood(self, sid: str, score: int, label: str = "", note: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO moods (session_id, score, label, note, ts) VALUES (?,?,?,?,?)",
                (sid, score, label, note, _now()),
            )
        return cur.lastrowid

    def moods(self, sid: str, limit: int = 30) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT score, label, note, ts FROM moods WHERE session_id=?"
                " ORDER BY id DESC LIMIT ?", (sid, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ---- journal --------------------------------------------------------
    def add_journal(self, sid: str, entry: str, prompt: str = "",
                    tags: Iterable[str] = ()) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO journal (session_id, prompt, entry, tags, ts) VALUES (?,?,?,?,?)",
                (sid, prompt, entry, json.dumps(list(tags)), _now()),
            )
        return cur.lastrowid

    def search_journal(self, sid: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT entry, prompt, tags, ts FROM journal WHERE session_id=?"
                " AND entry LIKE ? ORDER BY id DESC LIMIT ?",
                (sid, f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def journal_entries(self, sid: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT entry, prompt, tags, ts FROM journal WHERE session_id=?"
                " ORDER BY id DESC LIMIT ?", (sid, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- escalations (audit trail) --------------------------------------
    def log_escalation(self, sid: str, score: float, signals: List[str]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO escalations (session_id, risk_score, signals, ts)"
                " VALUES (?,?,?,?)",
                (sid, score, json.dumps(signals), _now()),
            )

    def escalations(self, sid: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT risk_score, signals, ts FROM escalations WHERE session_id=?"
                " ORDER BY id DESC", (sid,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- right to erasure ----------------------------------------------
    def wipe_session(self, sid: str) -> None:
        with self._conn() as c:
            for table in ("messages", "moods", "journal", "escalations"):
                c.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))
