"""
Embedding backends — 4-tier fallback chain.

Priority (each tier tried in order; first success wins)
────────────────────────────────────────────────────────
 1. SentenceTransformer   Best quality — pre-trained bi-encoder, 384-d
 2. HuggingFace AutoModel Mean-pooled transformer via `transformers` lib
 3. Word2Vec average       Lightweight, no GPU — gensim word vectors
 4. LSA (TF-IDF + SVD)    Zero-dependency fallback — worst semantic quality

Tier 1 is always preferred.  Tier 2 covers the case where sentence-transformers
has an import issue but PyTorch + transformers are available (e.g. version
mismatch).  Tier 3 uses pre-trained word vectors (50-d GloVe or Word2Vec) for
decent semantic similarity without GPU.  Tier 4 (LSA) is the absolute last
resort for fully offline, zero-model environments.

Offline model caching
─────────────────────
The factory calls _ensure_offline_cache() which downloads the model once to
data/models/ so subsequent runs work without internet.  Set env var
TRANSFORMERS_OFFLINE=1 to force offline mode after the first download.

All backends are wrapped with CachedEmbedder from src.embeddings so that
re-embedding on restart only happens for genuinely new text.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ── Tier 1: SentenceTransformer ──────────────────────────────────────────────

class _ST:
    """sentence-transformers bi-encoder — best quality."""

    def __init__(self, model, dim: int, model_name: str = ""):
        self._m = model
        self._d = dim
        self.model_name = model_name
        self.tier = 1

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array(
            self._m.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def fit(self, texts):
        pass  # pre-trained

    @property
    def dim(self) -> int:
        return self._d


# ── Tier 2: HuggingFace AutoModel mean-pooling ──────────────────────────────

class _HFTransformer:
    """transformers AutoModel with mean pooling — same quality as ST."""

    def __init__(self, model_name: str, dim: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._d = dim
        self._torch = torch
        self.model_name = model_name
        self.tier = 2

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        import torch

        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**encoded)

        # Mean pooling over non-padding tokens
        attention_mask = encoded["attention_mask"]
        token_embeds = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        sum_embeds = (token_embeds * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        embeddings = sum_embeds / sum_mask

        # L2 normalize
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        embeddings = embeddings / norms
        return embeddings.cpu().numpy().astype(np.float32)

    def fit(self, texts):
        pass

    @property
    def dim(self) -> int:
        return self._d


# ── Tier 3: Word2Vec / GloVe average ─────────────────────────────────────────

class _WordVecAvg:
    """Average pre-trained word vectors — lightweight, decent quality."""

    def __init__(self, dim: int = 50):
        import gensim.downloader as api
        logger.info("Loading GloVe word vectors (glove-wiki-gigaword-%dd)…", dim)
        self._model = api.load(f"glove-wiki-gigaword-{dim}")
        self._d = dim
        self.model_name = f"glove-{dim}d"
        self.tier = 3

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for text in texts:
            tokens = text.lower().split()
            word_vecs = [self._model[w] for w in tokens if w in self._model]
            if word_vecs:
                avg = np.mean(word_vecs, axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-9)  # L2 normalize
            else:
                avg = np.zeros(self._d, dtype=np.float32)
            vecs.append(avg)
        return np.array(vecs, dtype=np.float32)

    def fit(self, texts):
        pass

    @property
    def dim(self) -> int:
        return self._d


# ── Tier 4: LSA fallback ─────────────────────────────────────────────────────

class LSABackend:
    """TF-IDF + TruncatedSVD pipeline — no network, no GPU."""

    def __init__(self, n_components: int = 128, store_path=None):
        self._req = n_components
        self._n = n_components
        self._path = Path(store_path or "data/lsa.pkl")
        self._corpus: list = []
        self._fitted = False
        self._pipeline = None
        self.model_name = "lsa"
        self.tier = 4
        self._load()
        logger.info("LSA backend ready (max_dim=%d)", n_components)

    def _make_pipe(self, n: int):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("svd",   TruncatedSVD(n_components=n, random_state=42)),
            ("norm",  Normalizer(copy=False)),
        ])

    def _load(self):
        if self._path.exists():
            with self._path.open("rb") as f:
                state = pickle.load(f)
            self._pipeline = state["pipe"]
            self._corpus = state["corpus"]
            self._n = state["n"]
            self._fitted = True
            logger.info("LSA loaded — %d docs, dim=%d", len(self._corpus), self._n)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as f:
            pickle.dump({"pipe": self._pipeline, "corpus": self._corpus, "n": self._n}, f)

    def fit(self, texts):
        seen = set(self._corpus)
        for t in texts:
            if t not in seen:
                self._corpus.append(t)
                seen.add(t)
        from sklearn.feature_extraction.text import TfidfVectorizer
        probe = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=1)
        probe.fit(self._corpus)
        vocab = len(probe.vocabulary_)
        self._n = min(self._req, max(1, vocab - 1))
        self._pipeline = self._make_pipe(self._n)
        self._pipeline.fit(self._corpus)
        self._fitted = True
        self._save()
        logger.info("LSA fitted — corpus=%d vocab=%d dim=%d", len(self._corpus), vocab, self._n)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not self._fitted:
            self.fit(texts)
        try:
            return self._pipeline.transform(texts).astype(np.float32)
        except Exception:
            self.fit(texts)
            return self._pipeline.transform(texts).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._n


# ── Offline model caching ────────────────────────────────────────────────────

def _ensure_offline_cache(model_name: str, cache_root: str = "data/models") -> str:
    """
    Download and cache the model locally so it works offline on next run.

    Returns the local path if cached, or the original model_name for
    first-time online download.
    """
    cache_dir = Path(cache_root) / model_name.replace("/", "__")
    if cache_dir.exists() and any(cache_dir.iterdir()):
        logger.info("Using cached model from %s", cache_dir)
        return str(cache_dir)

    # Try to download and save locally
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        m.save(str(cache_dir))
        logger.info("Model cached to %s for offline use", cache_dir)
        return str(cache_dir)
    except Exception:
        pass
    return model_name


# ── factory ──────────────────────────────────────────────────────────────────

def build_embedder(model_name: str = "", lsa_dim: int = 128, lsa_path=None,
                   cache_dir=None, use_cache: bool = True):
    """
    Build the embedding backend (4-tier fallback) and wrap with disk KV cache.
    """
    raw = _build_raw(model_name, lsa_dim, lsa_path)
    tier_names = {1: "SentenceTransformer", 2: "HF AutoModel",
                  3: "Word2Vec/GloVe", 4: "LSA"}
    logger.info("Embedding tier: %d (%s)  dim=%d",
                raw.tier, tier_names.get(raw.tier, "?"), raw.dim)
    if use_cache and cache_dir:
        from src.embeddings import CachedEmbedder
        logger.info("Wrapping embedder with KV cache at %s", cache_dir)
        return CachedEmbedder(raw, cache_dir=str(cache_dir))
    return raw


def _build_raw(model_name: str, lsa_dim: int, lsa_path):
    """Try each tier in order; return the first that succeeds."""

    # Tier 1: SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
        local_path = _ensure_offline_cache(model_name)
        m = SentenceTransformer(local_path)
        d = (m.get_embedding_dimension()
             if hasattr(m, "get_embedding_dimension")
             else m.get_sentence_embedding_dimension())
        logger.info("Tier 1 — SentenceTransformer loaded: %s  dim=%d", model_name, d)
        return _ST(m, d, model_name)
    except Exception as e:
        logger.warning("Tier 1 SentenceTransformer failed: %s", e)

    # Tier 2: HuggingFace AutoModel (mean pooling)
    try:
        from transformers import AutoModel, AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        dim = cfg.hidden_size
        backend = _HFTransformer(model_name, dim)
        # Smoke test
        _ = backend.encode(["test"])
        logger.info("Tier 2 — HF AutoModel loaded: %s  dim=%d", model_name, dim)
        return backend
    except Exception as e:
        logger.warning("Tier 2 HF AutoModel failed: %s", e)

    # Tier 3: Word2Vec/GloVe average
    try:
        backend = _WordVecAvg(dim=50)
        _ = backend.encode(["test"])
        logger.info("Tier 3 — GloVe word vectors loaded  dim=50")
        return backend
    except Exception as e:
        logger.warning("Tier 3 Word2Vec/GloVe failed: %s", e)

    # Tier 4: LSA (always works — sklearn only)
    logger.warning("All transformer tiers failed — using Tier 4 LSA fallback")
    return LSABackend(n_components=lsa_dim, store_path=lsa_path)
