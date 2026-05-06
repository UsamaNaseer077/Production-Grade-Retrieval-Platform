"""
Test suite — 20 tests covering all platform layers.

Run:  pytest tests/ -v
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import Document, HashStore, IngestPipeline
from src.ingestion.chunker import Chunker
from src.indexing.embedder import LSABackend
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever, CrossEncoderReranker, _rrf
from src.embeddings.cache import EmbeddingCache
from src.embeddings import CachedEmbedder
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.generation.prompt_manager import PromptManager
from src.generation.self_check import SelfChecker
from src.telemetry.latency import LatencyTracker

CORPUS = [
    "BM25 is a sparse retrieval method based on term frequency and document frequency.",
    "Dense retrieval encodes queries and documents as embedding vectors.",
    "Hybrid retrieval combines sparse BM25 and dense vector search via fusion.",
    "Reciprocal Rank Fusion merges ranked lists without needing score normalisation.",
    "A cross-encoder re-ranker scores query and document jointly for higher precision.",
    "Chunking splits documents into smaller pieces for more precise retrieval.",
    "Parent-child chunking embeds small chunks but returns larger parent context.",
    "Incremental indexing skips unchanged files using a content hash store.",
    "Overfitting occurs when a model memorises training data but fails to generalise.",
    "Gradient descent iteratively updates model weights to minimise the loss function.",
]


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def doc():
    return Document("d1", "test.txt", "\n".join(CORPUS), {"filename": "test.txt"})


@pytest.fixture
def chunks(doc):
    return Chunker("parent_child", 50, 200, 5).chunk(doc)


# ── Ingestion ─────────────────────────────────────────────────────────────────

def test_incremental_skip(tmp):
    f = tmp / "doc.txt"
    f.write_text("hello world this is a test document")
    hs = HashStore(tmp / "h.json")
    p = IngestPipeline(hs)
    assert len(p.ingest_file(f)) == 1
    assert len(p.ingest_file(f)) == 0        # unchanged → skip
    f.write_text("hello world MODIFIED content")
    assert len(p.ingest_file(f)) == 1        # changed → re-ingest


def test_csv_row_granularity(tmp):
    f = tmp / "data.csv"
    with f.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "v"])
        w.writerows([["a", "alpha"], ["b", "beta"], ["c", "gamma"]])
    hs = HashStore(tmp / "h.json")
    assert len(IngestPipeline(hs).ingest_file(f)) == 3


def test_unsupported_skipped(tmp):
    f = tmp / "img.png"
    f.write_bytes(b"\x89PNG")
    hs = HashStore(tmp / "h.json")
    assert IngestPipeline(hs).ingest_file(f) == []


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunks_produced(chunks):
    assert len(chunks) > 0


def test_chunk_ids_unique(chunks):
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_parent_gte_child(chunks):
    assert all(len(c.parent_text) >= len(c.text) for c in chunks)


def test_all_strategies(doc):
    for s in ("fixed", "sentence", "parent_child", "sliding", "structural"):
        result = Chunker(s, 30, 120, 5).chunk(doc)
        assert len(result) > 0, f"Strategy '{s}' produced 0 chunks"


# ── Embedding cache ───────────────────────────────────────────────────────────

def test_embedding_cache_hit_miss(tmp):
    cache = EmbeddingCache(cache_dir=str(tmp / "cache"), model_tag="test")
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert cache.get("hello") is None
    cache.set("hello", vec)
    loaded = cache.get("hello")
    assert loaded is not None
    assert np.allclose(loaded, vec)
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1


def test_cached_embedder_avoids_double_encode(tmp):
    lsa = LSABackend(n_components=8, store_path=tmp / "lsa.pkl")
    lsa.fit(CORPUS)
    wrapped = CachedEmbedder(lsa, cache_dir=str(tmp / "emb_cache"))
    v1 = wrapped.encode(["BM25 sparse retrieval"])
    v2 = wrapped.encode(["BM25 sparse retrieval"])  # should hit cache
    assert np.allclose(v1, v2)
    assert wrapped.cache_stats["hits"] >= 1


# ── Embedder ──────────────────────────────────────────────────────────────────

def test_lsa_encodes(tmp):
    lsa = LSABackend(n_components=32, store_path=tmp / "lsa.pkl")
    lsa.fit(CORPUS)
    vecs = lsa.encode(["sparse retrieval BM25"])
    assert vecs.shape[0] == 1
    assert lsa.dim >= 1


# ── Vector store ──────────────────────────────────────────────────────────────

def test_vs_add_search(tmp, chunks):
    lsa = LSABackend(16, tmp / "lsa.pkl")
    lsa.fit([c.text for c in chunks])
    vs = VectorStore(
        lsa_dim=16, index_path=tmp / "f.index",
        meta_path=tmp / "m.jsonl", lsa_path=tmp / "lsa.pkl", embedder=lsa,
    )
    assert vs.add_chunks(chunks) == len(chunks)
    assert vs.add_chunks(chunks) == 0          # idempotent
    res = vs.search("hybrid retrieval BM25", top_k=3)
    assert len(res) > 0 and "chunk_id" in res[0]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def test_bm25_add_search(tmp, chunks):
    bm = BM25Store(tmp / "bm.pkl")
    assert bm.add_chunks(chunks) == len(chunks)
    assert bm.add_chunks(chunks) == 0
    res = bm.search("BM25 sparse term frequency", top_k=5)
    assert len(res) > 0


# ── Fusion ────────────────────────────────────────────────────────────────────

def test_rrf_overlap_wins():
    dense  = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "b", "text": "b"}]
    sparse = [{"chunk_id": "b", "text": "b"}, {"chunk_id": "c", "text": "c"}]
    ranked, _ = _rrf(dense, sparse)
    assert ranked[0][0] == "b"   # b appears in both lists → highest RRF score
    assert all(sc > 0 for _, sc in ranked)  # RRF scores are always positive


# ── Memory ────────────────────────────────────────────────────────────────────

def test_short_term_memory():
    stm = ShortTermMemory(max_turns=3)
    stm.add("s1", "user", "What is BM25?")
    stm.add("s1", "assistant", "BM25 is a ranking function.")
    msgs = stm.get("s1")
    assert len(msgs) == 2
    formatted = stm.format_for_prompt("s1")
    assert "BM25" in formatted


def test_long_term_memory(tmp):
    ltm = LongTermMemory(db_path=str(tmp / "mem.db"))
    ltm.store("sess1", "What is BM25?", "BM25 is a ranking function.", ["chunk1"])
    ltm.store("sess1", "What is RRF?", "RRF merges ranked lists.", ["chunk2"])
    assert ltm.count() == 2
    results = ltm.search("BM25 ranking")
    assert len(results) >= 1
    assert "BM25" in results[0].question


# ── Prompt manager ────────────────────────────────────────────────────────────

def test_prompt_manager_default(tmp):
    pm = PromptManager(prompt_path=str(tmp / "nonexistent.md"))
    assert len(pm.system_prompt) > 10
    prompt = pm.build(
        query="What is hybrid retrieval?",
        contexts=[{"source": "test.txt", "text": "Hybrid combines BM25 and dense.",
                   "parent_text": "Hybrid combines BM25 and dense."}],
    )
    assert "CONTEXT" in prompt
    assert "QUESTION" in prompt


def test_prompt_manager_custom(tmp):
    md = tmp / "system.md"
    md.write_text("You are a test bot.")
    pm = PromptManager(prompt_path=str(md))
    assert pm.system_prompt == "You are a test bot."


# ── Self-check ────────────────────────────────────────────────────────────────

def test_self_check_grounded():
    checker = SelfChecker(min_overlap=0.05)
    contexts = [{"parent_text": "BM25 is a sparse retrieval method using term frequency."}]
    result = checker.check("BM25 uses term frequency for ranking.", contexts)
    assert result["grounded"] is True
    assert result["score"] > 0


def test_self_check_ungrounded():
    checker = SelfChecker(min_overlap=0.3)
    contexts = [{"parent_text": "The sky is blue on a clear day."}]
    result = checker.check("Quantum mechanics describes subatomic particles.", contexts)
    assert result["score"] < 0.3


# ── Latency tracker ───────────────────────────────────────────────────────────

def test_latency_tracker():
    tracker = LatencyTracker()
    import time
    with tracker.track("dense"):
        time.sleep(0.01)
    with tracker.track("dense"):
        time.sleep(0.01)
    s = tracker.stats("dense")
    assert s["count"] == 2
    assert s["mean_ms"] >= 10
    assert "p50_ms" in s


# ── End-to-end ────────────────────────────────────────────────────────────────

def test_e2e(tmp, chunks):
    lsa = LSABackend(16, tmp / "lsa.pkl")
    lsa.fit([c.text for c in chunks])
    vs  = VectorStore(
        lsa_dim=16, index_path=tmp / "f.index",
        meta_path=tmp / "m.jsonl", lsa_path=tmp / "lsa.pkl", embedder=lsa,
    )
    bm  = BM25Store(tmp / "bm.pkl")
    vs.add_chunks(chunks)
    bm.add_chunks(chunks)
    r   = HybridRetriever(vs, bm, CrossEncoderReranker(), top_k_rerank=3)
    res = r.search("hybrid retrieval dense sparse BM25")
    assert len(res) > 0 and res[0].latency_ms > 0


# ── Eval harness ──────────────────────────────────────────────────────────────

def test_eval(tmp, chunks):
    lsa = LSABackend(16, tmp / "lsa.pkl")
    lsa.fit([c.text for c in chunks])
    vs  = VectorStore(
        lsa_dim=16, index_path=tmp / "f.index",
        meta_path=tmp / "m.jsonl", lsa_path=tmp / "lsa.pkl", embedder=lsa,
    )
    bm  = BM25Store(tmp / "bm.pkl")
    vs.add_chunks(chunks)
    bm.add_chunks(chunks)
    r = HybridRetriever(vs, bm, CrossEncoderReranker(), top_k_rerank=3)
    first_id = chunks[0].chunk_id
    (tmp / "q.json").write_text(json.dumps([{"query_id": "q1", "text": "BM25 sparse retrieval"}]))
    (tmp / "qr.json").write_text(json.dumps({"q1": [first_id]}))
    from src.evaluation.harness import run_eval
    rep = run_eval(r, tmp / "q.json", tmp / "qr.json", [1, 3])
    assert "mrr" in rep and "recall@1" in rep
    assert rep["latency"]["mean_ms"] >= 0
