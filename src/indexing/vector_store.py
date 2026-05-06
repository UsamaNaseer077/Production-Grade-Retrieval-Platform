import json, logging, time
from pathlib import Path
from typing import Optional
import faiss, numpy as np
from src.indexing.embedder import build_embedder

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name="", lsa_dim=128, hnsw_m=32,
                 index_path=None, meta_path=None, lsa_path=None, embedder=None):
        self.embedder   = embedder or build_embedder(model_name, lsa_dim, lsa_path)
        self.index_path = Path(index_path or "data/faiss.index")
        self.meta_path  = Path(meta_path  or "data/meta.jsonl")
        self._hnsw_m    = hnsw_m
        self._meta: list[dict] = []
        self._ids:  set[str]   = set()
        self._index = None
        self._load()

    def _new_index(self, dim):
        idx = faiss.IndexHNSWFlat(dim, self._hnsw_m)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch       = 64
        return idx

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with self.meta_path.open() as f:
                self._meta = [json.loads(l) for l in f if l.strip()]
            self._ids = {m["chunk_id"] for m in self._meta}
            logger.info("Loaded index — %d vectors", self._index.ntotal)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w") as f:
            for m in self._meta: f.write(json.dumps(m)+"\n")
        logger.info("Saved index — %d vectors", self._index.ntotal)

    def add_chunks(self, chunks):
        new = [c for c in chunks if c.chunk_id not in self._ids]
        if not new: return 0
        texts = [c.text for c in new]
        if hasattr(self.embedder, "fit"): self.embedder.fit(texts)
        vecs = self.embedder.encode(texts)
        dim  = vecs.shape[1]
        if self._index is None or self._index.d != dim:
            # rebuild with correct dim
            self._index = self._new_index(dim)
            if self._meta:
                old_vecs = self.embedder.encode([m["text"] for m in self._meta])
                self._index.add(old_vecs)
        self._index.add(vecs)
        for c in new:
            self._meta.append({"chunk_id":c.chunk_id,"doc_id":c.doc_id,
                "parent_id":c.parent_id,"text":c.text,"parent_text":c.parent_text,
                "source":c.source,"metadata":c.metadata})
            self._ids.add(c.chunk_id)
        logger.info("Indexed %d chunks. Total: %d", len(new), self._index.ntotal)
        return len(new)

    def search(self, query, top_k=50):
        if self._index is None or self._index.ntotal == 0: return []
        t0 = time.perf_counter()
        q  = self.embedder.encode([query])
        D, I = self._index.search(q, min(top_k, self._index.ntotal))
        lat  = (time.perf_counter()-t0)*1000
        return [{**self._meta[i],"dense_score":float(d),"dense_rank":r,"latency_ms":lat}
                for r,(d,i) in enumerate(zip(D[0],I[0])) if i>=0]

    @property
    def size(self): return 0 if self._index is None else self._index.ntotal
