import logging, pickle, re, time
from pathlib import Path
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def _tok(t):
    t = re.sub(r"[^\w\s]"," ",t.lower()); return t.split()

class BM25Store:
    def __init__(self, store_path=None):
        self._path  = Path(store_path or "data/bm25.pkl")
        self._bm25  = None
        self._meta: list[dict] = []
        self._ids:  set[str]   = set()
        self._load()

    def _load(self):
        if self._path.exists():
            with self._path.open("rb") as f: s = pickle.load(f)
            self._bm25 = s["bm25"]; self._meta = s["meta"]
            self._ids  = {m["chunk_id"] for m in self._meta}
            logger.info("BM25 loaded — %d docs", len(self._meta))

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as f:
            pickle.dump({"bm25":self._bm25,"meta":self._meta}, f)
        logger.info("BM25 saved — %d docs", len(self._meta))

    def add_chunks(self, chunks):
        new = [c for c in chunks if c.chunk_id not in self._ids]
        if not new: return 0
        for c in new:
            self._meta.append({"chunk_id":c.chunk_id,"doc_id":c.doc_id,
                "parent_id":c.parent_id,"text":c.text,"parent_text":c.parent_text,
                "source":c.source,"metadata":c.metadata})
            self._ids.add(c.chunk_id)
        self._bm25 = BM25Okapi([_tok(m["text"]) for m in self._meta])
        logger.info("BM25 rebuilt — %d docs", len(self._meta))
        return len(new)

    def search(self, query, top_k=50):
        if not self._bm25: return []
        t0     = time.perf_counter()
        scores = self._bm25.get_scores(_tok(query))
        lat    = (time.perf_counter()-t0)*1000
        ranked = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)
        return [{**self._meta[i],"sparse_score":float(s),"sparse_rank":r,"latency_ms":lat}
                for r,(i,s) in enumerate(ranked[:top_k])]

    @property
    def size(self): return len(self._meta)
