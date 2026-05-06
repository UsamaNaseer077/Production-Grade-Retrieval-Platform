"""
Long-term (episodic + semantic) memory — SQLite-backed Q&A store.

Stores every answered question with its sources.  Provides:
  - keyword search over stored Q&As (relevant past context injection)
  - session history retrieval
  - decay: old records that haven't been accessed can be pruned

For million-scale deployments: swap SQLite for PostgreSQL + pgvector.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    id: int
    session_id: str
    question: str
    answer: str
    source_ids: List[str]
    timestamp: float

    @property
    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600


class LongTermMemory:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── schema ───────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT    NOT NULL,
                    question   TEXT    NOT NULL,
                    answer     TEXT    NOT NULL,
                    source_ids TEXT    DEFAULT '[]',
                    timestamp  REAL    NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_sid ON memory(session_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ts  ON memory(timestamp)")
            # FTS for keyword search
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(question, answer, content='memory', content_rowid='id')
            """)
            # keep FTS in sync via triggers
            c.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memory BEGIN
                    INSERT INTO memory_fts(rowid, question, answer)
                    VALUES (new.id, new.question, new.answer);
                END
            """)
            c.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memory BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, question, answer)
                    VALUES ('delete', old.id, old.question, old.answer);
                END
            """)

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── writes ───────────────────────────────────────────────────────────────

    def store(
        self,
        session_id: str,
        question: str,
        answer: str,
        source_ids: Optional[List[str]] = None,
    ):
        with self._conn() as c:
            c.execute(
                "INSERT INTO memory(session_id,question,answer,source_ids,timestamp)"
                " VALUES (?,?,?,?,?)",
                (session_id, question, answer, json.dumps(source_ids or []), time.time()),
            )
        logger.debug("LTM stored Q&A for session %s", session_id)

    def prune_old(self, older_than_hours: float = 720):
        """Remove records older than N hours (default 30 days)."""
        cutoff = time.time() - older_than_hours * 3600
        with self._conn() as c:
            deleted = c.execute(
                "DELETE FROM memory WHERE timestamp < ?", (cutoff,)
            ).rowcount
        if deleted:
            logger.info("LTM pruned %d old records", deleted)


    # ── reads ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fts_query(raw: str) -> str:
        """
        Convert a free-text query to a safe FTS5 MATCH expression.

        FTS5 MATCH syntax is strict: bare spaces mean AND, special chars
        (like '-', '(', ')') cause OperationalError.  We build an OR query
        so that ANY query token produces a hit, which works well for fuzzy
        memory recall.

        Example:
          "rank fusion merging lists" → '"rank" OR "fusion" OR "merging" OR "lists"'
        """
        import re as _re
        # remove FTS5 special chars: ( ) * : ^ " -
        clean = _re.sub(r'[()\*:\^"\-]', ' ', raw)
        tokens = [t for t in clean.lower().split() if len(t) > 2]
        if not tokens:
            # worst-case: match everything
            return '""'
        return " OR ".join(f'"{t}"' for t in tokens)

    def search(
        self,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """FTS keyword search. Falls back to multi-term LIKE if FTS table empty."""
        fts_q = self._fts_query(query)
        with self._conn() as c:
            # try FTS first
            try:
                if session_id:
                    rows = c.execute(
                        "SELECT m.id,m.session_id,m.question,m.answer,m.source_ids,m.timestamp"
                        " FROM memory_fts f JOIN memory m ON m.id=f.rowid"
                        " WHERE memory_fts MATCH ? AND m.session_id=?"
                        " ORDER BY rank LIMIT ?",
                        (fts_q, session_id, limit),
                    ).fetchall()
                else:
                    rows = c.execute(
                        "SELECT m.id,m.session_id,m.question,m.answer,m.source_ids,m.timestamp"
                        " FROM memory_fts f JOIN memory m ON m.id=f.rowid"
                        " WHERE memory_fts MATCH ?"
                        " ORDER BY rank LIMIT ?",
                        (fts_q, limit),
                    ).fetchall()
            except Exception:  # FTS table empty, corrupted, or parse error
                # LIKE fallback: search both question and answer columns
                terms = query.lower().split()[:5]
                cond_parts = ["(lower(question) LIKE ? OR lower(answer) LIKE ?)" for _ in terms]
                params: list = []
                for t in terms:
                    params += [f"%{t}%", f"%{t}%"]
                extra = ""
                if session_id:
                    extra = " AND session_id=?"
                    params.append(session_id)
                params.append(limit)
                rows = c.execute(
                    f"SELECT id,session_id,question,answer,source_ids,timestamp"
                    f" FROM memory WHERE ({' OR '.join(cond_parts)}){extra}"
                    f" ORDER BY timestamp DESC LIMIT ?",
                    params,
                ).fetchall()
        return [
            MemoryRecord(r[0], r[1], r[2], r[3], json.loads(r[4]), r[5]) for r in rows
        ]

    def get_session_history(
        self, session_id: str, limit: int = 20
    ) -> List[MemoryRecord]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,session_id,question,answer,source_ids,timestamp"
                " FROM memory WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return list(
            reversed(
                [MemoryRecord(r[0], r[1], r[2], r[3], json.loads(r[4]), r[5]) for r in rows]
            )
        )

    def format_relevant(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_chars: int = 1200,
    ) -> str:
        """Return formatted past Q&As relevant to query (for prompt injection)."""
        records = self.search(query, limit=3, session_id=session_id)
        if not records:
            return ""
        parts = [f"Q: {r.question}\nA: {r.answer}" for r in records]
        text = "\n\n".join(parts)
        return text[:max_chars]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
