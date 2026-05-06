"""
Disk-based KV cache for embeddings.

Layout:  cache_dir/<shard_2_chars>/<sha256_hex>.npy
Each file is a single (dim,) float32 numpy array.
Sharding prevents single-directory inode exhaustion at 1M+ entries.
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, cache_dir: str = "data/emb_cache", model_tag: str = ""):
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._tag = model_tag
        self._hits = 0
        self._misses = 0

    # ── key / path ──────────────────────────────────────────────────────────

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._tag}:{text}".encode("utf-8", errors="replace")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.npy"

    # ── single-item ops ─────────────────────────────────────────────────────

    def get(self, text: str) -> Optional[np.ndarray]:
        p = self._path(self._key(text))
        if p.exists():
            try:
                vec = np.load(str(p))
                self._hits += 1
                return vec
            except Exception:
                p.unlink(missing_ok=True)
        self._misses += 1
        return None

    def set(self, text: str, vec: np.ndarray):
        key = self._key(text)
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), vec.astype(np.float32))

    # ── batch ops ───────────────────────────────────────────────────────────

    def get_batch(
        self, texts: List[str]
    ) -> Tuple[List[Optional[np.ndarray]], List[str], List[int]]:
        """
        Returns:
            result       – list of arrays or None for each input text
            miss_texts   – texts not in cache
            miss_indices – their positions in `texts`
        """
        result: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_indices: List[int] = []
        for i, text in enumerate(texts):
            vec = self.get(text)
            if vec is not None:
                result[i] = vec
            else:
                miss_texts.append(text)
                miss_indices.append(i)
        return result, miss_texts, miss_indices

    def set_batch(self, texts: List[str], vecs: np.ndarray):
        for text, vec in zip(texts, vecs):
            self.set(text, vec)

    # ── stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "cached_files": sum(1 for _ in self._root.rglob("*.npy")),
        }
