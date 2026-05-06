"""
Embedding layer with disk-based KV cache.

CachedEmbedder wraps any encoder (SentenceTransformer or LSA) and caches
every computed vector to disk keyed by sha256(model+text).  On restart,
only texts that were never seen before are re-encoded — cold-start cost
drops to near-zero for large corpora that are already indexed.
"""
from .cache import EmbeddingCache
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CachedEmbedder:
    """Drop-in wrapper that adds KV caching to any embedder."""

    def __init__(self, embedder, cache_dir="data/emb_cache"):
        self._enc = embedder
        model_id = getattr(embedder, "model_name", "") or getattr(
            getattr(embedder, "_m", None), "model_card_data", {}).get("base_model", "")
        self._cache = EmbeddingCache(cache_dir=cache_dir, model_tag=str(model_id))

    # ── public API ──────────────────────────────────────────────────────────

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result, miss_texts, miss_idx = self._cache.get_batch(texts)
        if miss_texts:
            new_vecs = self._enc.encode(miss_texts)
            if new_vecs.ndim == 1:
                new_vecs = new_vecs.reshape(1, -1)
            self._cache.set_batch(miss_texts, new_vecs)
            for arr_idx, orig_idx in enumerate(miss_idx):
                result[orig_idx] = new_vecs[arr_idx]
        return np.array(result, dtype=np.float32)

    def fit(self, texts):
        if hasattr(self._enc, "fit"):
            self._enc.fit(texts)

    @property
    def dim(self):
        return self._enc.dim

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats
