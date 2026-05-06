"""
Chunking strategies.

Strategy       When to use
-----------    -------------------------------------------------------
fixed          Baseline; fast; ignores sentence boundaries
sentence       Better coherence; sentence-boundary aware
parent_child   Recommended: embed 256-token child, return 1 024-token
               parent for coherent context delivery (default)
sliding        Overlapping windows; good for dense retrieval on
               long documents with subtle phrasing
structural     Splits on Markdown headings; best for docling output
               where heading structure has been preserved
"""
import re
from dataclasses import dataclass, field
from typing import List


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    parent_text: str
    source: str
    metadata: dict = field(default_factory=dict)
    child_index: int = 0
    parent_index: int = 0


# ── primitive splitters ───────────────────────────────────────────────────────

def _words(t: str) -> List[str]:
    return t.split()


def _sentences(t: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def _headings(t: str) -> List[str]:
    """Split on Markdown H1–H3 boundaries, keeping the heading with its body."""
    parts = re.split(r"(?=^#{1,3} )", t, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _fixed_windows(text: str, size: int, overlap: int) -> List[str]:
    words = _words(text)
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + size]))
        i += size - overlap
    return chunks


def _sent_windows(text: str, size: int, overlap: int) -> List[str]:
    sents = _sentences(text)
    chunks, buf, buf_len = [], [], 0
    for s in sents:
        wl = len(s.split())
        if buf_len + wl > size and buf:
            chunks.append(" ".join(buf))
            # carry-over overlap (in words)
            tail = " ".join(buf).split()[-overlap:]
            buf, buf_len = (([" ".join(tail)] if tail else []), len(tail))
        buf.append(s)
        buf_len += wl
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# ── chunker ───────────────────────────────────────────────────────────────────

class Chunker:
    def __init__(
        self,
        strategy: str = "parent_child",
        child_size: int = 256,
        parent_size: int = 1024,
        overlap: int = 32,
    ):
        self.strategy = strategy
        self.cs = child_size
        self.ps = parent_size
        self.ov = overlap

    # ── dispatch ─────────────────────────────────────────────────────────────

    def chunk(self, doc) -> List[Chunk]:
        dispatch = {
            "fixed":        self._fixed,
            "sentence":     self._sentence,
            "parent_child": self._parent_child,
            "sliding":      self._sliding,
            "structural":   self._structural,
        }
        fn = dispatch.get(self.strategy, self._parent_child)
        return fn(doc)

    def chunk_many(self, docs) -> List[Chunk]:
        out = []
        for d in docs:
            out.extend(self.chunk(d))
        return out

    # ── helpers ───────────────────────────────────────────────────────────────

    def _mk(self, doc, pi: int, ci: int, child: str, parent: str) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}_{pi}_{ci}",
            doc_id=doc.doc_id,
            parent_id=f"{doc.doc_id}_{pi}",
            text=child,
            parent_text=parent,
            source=doc.source,
            metadata=doc.metadata,
            child_index=ci,
            parent_index=pi,
        )

    # ── strategies ────────────────────────────────────────────────────────────

    def _fixed(self, doc) -> List[Chunk]:
        texts = _fixed_windows(doc.text, self.cs, self.ov)
        return [self._mk(doc, i, i, t, t) for i, t in enumerate(texts)]

    def _sentence(self, doc) -> List[Chunk]:
        texts = _sent_windows(doc.text, self.cs, self.ov)
        return [self._mk(doc, i, i, t, t) for i, t in enumerate(texts)]

    def _parent_child(self, doc) -> List[Chunk]:
        """Embed small child chunks; return their larger parent for context."""
        parents = _sent_windows(doc.text, self.ps, self.ov)
        chunks = []
        for pi, pt in enumerate(parents):
            for ci, ct in enumerate(_sent_windows(pt, self.cs, self.ov)):
                chunks.append(self._mk(doc, pi, ci, ct, pt))
        return chunks

    def _sliding(self, doc) -> List[Chunk]:
        """
        Overlapping fixed windows.  Step = child_size - overlap.
        Higher recall on long documents at the cost of index size.
        """
        words = _words(doc.text)
        step = max(1, self.cs - self.ov)
        chunks = []
        for i, start in enumerate(range(0, max(1, len(words) - self.cs + 1), step)):
            ct = " ".join(words[start: start + self.cs])
            # parent = a larger centred window for context delivery
            p_start = max(0, start - self.ov)
            pt = " ".join(words[p_start: start + self.cs + self.ov])
            chunks.append(self._mk(doc, i, i, ct, pt))
        return chunks

    def _structural(self, doc) -> List[Chunk]:
        """
        Split on Markdown heading boundaries (docling output).
        Each section becomes a parent; its sub-sentences become children.
        Falls back to parent_child when no headings found.
        """
        sections = _headings(doc.text)
        if len(sections) <= 1:
            return self._parent_child(doc)
        chunks = []
        for pi, pt in enumerate(sections):
            for ci, ct in enumerate(_sent_windows(pt, self.cs, self.ov)):
                chunks.append(self._mk(doc, pi, ci, ct, pt))
        return chunks
