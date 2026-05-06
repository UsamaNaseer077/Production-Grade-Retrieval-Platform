"""
Prompt manager — loads the system persona from prompts/system.md and
assembles the full RAG prompt for each query.

The .md file supports a YAML-like front-matter block that is stripped
before the prompt is sent to the model.
"""
import re
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = """\
You are a precise, knowledgeable retrieval assistant.

Rules:
- Answer ONLY from the provided CONTEXT.
- If the context does not contain enough information, say so explicitly.
- Do not fabricate facts, statistics, or citations.
- Keep answers concise and factual.
- When relevant, cite source numbers like [1] or [2].
"""


class PromptManager:
    def __init__(self, prompt_path: str = "prompts/system.md"):
        self._path = Path(prompt_path)
        self._system = self._load()

    def _load(self) -> str:
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8")
            # Strip YAML front-matter (---...---) if present
            raw = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL).strip()
            if raw:
                logger.info("Loaded system prompt from %s", self._path)
                return raw
        logger.info("prompts/system.md not found — using built-in default")
        return _DEFAULT_SYSTEM

    def reload(self):
        """Hot-reload the prompt file without restarting the server."""
        self._system = self._load()

    @property
    def system_prompt(self) -> str:
        return self._system

    # ── prompt assembly ──────────────────────────────────────────────────────

    def build(
        self,
        query: str,
        contexts: List[dict],
        conversation_history: str = "",
        memory_context: str = "",
        grounding_reminder: bool = False,
    ) -> str:
        """
        Build the user-turn prompt that goes after the system prompt.

        Sections (included only if non-empty):
          CONTEXT            — retrieved passages
          RELEVANT PAST Q&As — long-term memory hits
          CONVERSATION       — short-term session history
          QUESTION           — the user query
        """
        parts: List[str] = []

        # ── retrieved context ────────────────────────────────────────────────
        ctx_blocks = []
        for i, c in enumerate(contexts, 1):
            src = c.get("source", "unknown")
            text = c.get("parent_text", c.get("text", ""))[:800]  # token budget
            ctx_blocks.append(f"[{i}] ({src})\n{text}")
        parts.append("CONTEXT:\n" + ("\n\n".join(ctx_blocks) if ctx_blocks else "None."))

        # ── long-term memory ─────────────────────────────────────────────────
        if memory_context:
            parts.append(f"RELEVANT PAST Q&As:\n{memory_context}")

        # ── short-term conversation ──────────────────────────────────────────
        if conversation_history:
            parts.append(f"CONVERSATION HISTORY:\n{conversation_history}")

        # ── question ─────────────────────────────────────────────────────────
        parts.append(f"QUESTION: {query}")

        if grounding_reminder:
            parts.append(
                "REMINDER: If the context does not contain a clear answer, "
                "respond with: 'The available context does not contain sufficient "
                "information to answer this question confidently.'"
            )

        return "\n\n".join(parts)
