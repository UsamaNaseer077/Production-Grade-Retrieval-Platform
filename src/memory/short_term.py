"""
Short-term (episodic) memory — per-session conversation ring buffer.

Keeps the last N turns in RAM.  Stale sessions are evicted after TTL
seconds.  No external dependency required; swap _sessions to Redis for
multi-process deployments.
"""
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    ts: float = field(default_factory=time.time)


class ShortTermMemory:
    def __init__(self, max_turns: int = 10, ttl_seconds: int = 3600):
        self.max_turns = max_turns
        self.ttl = ttl_seconds
        # maxlen = turns × 2 (user + assistant per turn)
        self._sessions: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_turns * 2)
        )
        self._last: Dict[str, float] = {}

    # ── writes ──────────────────────────────────────────────────────────────

    def add(self, session_id: str, role: str, content: str):
        self._evict()
        self._sessions[session_id].append(Message(role=role, content=content))
        self._last[session_id] = time.time()

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._last.pop(session_id, None)

    # ── reads ───────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> List[Message]:
        self._evict()
        return list(self._sessions.get(session_id, []))

    def as_dicts(self, session_id: str) -> List[dict]:
        return [{"role": m.role, "content": m.content} for m in self.get(session_id)]

    def format_for_prompt(self, session_id: str, max_chars: int = 2000) -> str:
        msgs = self.get(session_id)
        if not msgs:
            return ""
        lines = [
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in msgs
        ]
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    # ── housekeeping ─────────────────────────────────────────────────────────

    def _evict(self):
        now = time.time()
        stale = [sid for sid, t in self._last.items() if now - t > self.ttl]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last.pop(sid, None)
        if stale:
            logger.debug("Evicted %d stale STM sessions", len(stale))

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)
