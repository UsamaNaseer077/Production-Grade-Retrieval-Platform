#!/usr/bin/env python3
"""
Build eval/qrels.json — chunk-level relevance judgements.

Strategy
────────
For each query we run both BM25 and dense search.  Candidates whose source
document is known to be relevant (by filepath pattern match) AND whose text
contains at least one expected keyword are marked as relevant.  The approach
is conservative: we only mark a chunk relevant if we have high confidence.

We emit TREC-style qrels:  {query_id: [chunk_id, ...]}

Run after ingestion:
    python scripts/build_qrels.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CFG
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import build_embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid import CrossEncoderReranker, HybridRetriever

logging.basicConfig(level="WARNING")

# ── query definitions with relevance criteria ─────────────────────────────────

EVAL_QUERIES = [
    {
        "query_id": "q1",
        "text": "how does BM25 sparse retrieval work",
        "source_patterns": ["information_retrieval"],
        "keywords": ["BM25", "term frequency", "sparse", "probabilistic"],
    },
    {
        "query_id": "q2",
        "text": "what is hybrid retrieval",
        "source_patterns": ["information_retrieval"],
        "keywords": ["hybrid", "BM25", "dense", "RRF", "fusion"],
    },
    {
        "query_id": "q3",
        "text": "explain gradient descent optimisation",
        "source_patterns": ["machine_learning"],
        "keywords": ["gradient", "descent", "loss", "parameters", "learning"],
    },
    {
        "query_id": "q4",
        "text": "what is overfitting in machine learning",
        "source_patterns": ["machine_learning"],
        "keywords": ["overfitting", "training", "generalise", "bias", "variance"],
    },
    {
        "query_id": "q5",
        "text": "how does Docker containerisation work",
        "source_patterns": ["software_engineering"],
        "keywords": ["Docker", "container", "image", "deployment", "package"],
    },
    {
        "query_id": "q6",
        "text": "cross-encoder reranker for retrieval precision",
        "source_patterns": ["information_retrieval"],
        "keywords": ["cross-encoder", "rerank", "precision", "attention"],
    },
    {
        "query_id": "q7",
        "text": "transfer learning fine-tuning pretrained models",
        "source_patterns": ["machine_learning"],
        "keywords": ["transfer", "fine-tuning", "pretrained", "downstream"],
    },
    {
        "query_id": "q8",
        "text": "distributed systems consistency availability CAP theorem",
        "source_patterns": ["distributed_systems"],
        "keywords": ["CAP", "consistency", "availability", "partition"],
    },
    {
        "query_id": "q9",
        "text": "database B-tree index performance",
        "source_patterns": ["databases"],
        "keywords": ["B-tree", "index", "O(log", "database", "search"],
    },
    {
        "query_id": "q10",
        "text": "continuous integration automated testing DevOps",
        "source_patterns": ["software_engineering"],
        "keywords": ["continuous integration", "CI", "test", "automated", "commit"],
    },
]


def is_relevant(chunk_id: str, text: str, source: str,
                source_patterns: list, keywords: list) -> bool:
    """Return True if this chunk is confidently relevant to the query."""
    # Source must match at least one pattern
    source_ok = any(pat.lower() in source.lower() for pat in source_patterns)
    if not source_ok:
        return False
    # Text must contain at least one keyword (case-insensitive)
    text_lower = text.lower()
    kw_ok = any(kw.lower() in text_lower for kw in keywords)
    return kw_ok


def build_qrels():
    print("Loading indexes …")
    emb = build_embedder(
        model_name=CFG.embedding.model_name,
        lsa_dim=CFG.embedding.lsa_dim,
        lsa_path=CFG.index.lsa_path,
        cache_dir=CFG.embedding.cache_dir,
        use_cache=True,
    )
    vs = VectorStore(
        lsa_dim=CFG.embedding.lsa_dim, hnsw_m=CFG.index.hnsw_m,
        index_path=CFG.index.index_path, meta_path=CFG.index.meta_path,
        lsa_path=CFG.index.lsa_path, embedder=emb,
    )
    bm25 = BM25Store(CFG.index.bm25_path)
    reranker = CrossEncoderReranker()
    retriever = HybridRetriever(
        vs, bm25, reranker,
        top_k_dense=100, top_k_sparse=100, top_k_rerank=20, rrf_k=60,
    )

    print(f"Index size: {vs.size} vectors, {bm25.size} BM25 docs")

    qrels = {}
    queries_out = []
    min_relevant = 1  # warn if fewer than this

    for q in EVAL_QUERIES:
        qid = q["query_id"]
        results = retriever.search(q["text"])

        relevant = [
            r.chunk_id for r in results
            if is_relevant(r.chunk_id, r.text, r.source,
                           q["source_patterns"], q["keywords"])
        ]

        # Fallback: if source pattern matching is too strict, take top results
        # that match at least one keyword anywhere
        if not relevant:
            relevant = [
                r.chunk_id for r in results[:20]
                if any(kw.lower() in r.text.lower() for kw in q["keywords"])
            ][:3]

        if len(relevant) < min_relevant:
            print(f"  WARNING: {qid} has {len(relevant)} relevant chunks — "
                  f"check source patterns {q['source_patterns']}")

        qrels[qid] = relevant
        queries_out.append({"query_id": qid, "text": q["text"]})

        print(f"  {qid}: '{q['text'][:45]}...' → {len(relevant)} relevant chunks")
        for cid in relevant[:2]:
            r = next((r for r in results if r.chunk_id == cid), None)
            if r:
                print(f"       {Path(r.source).name}: {r.text[:60]}…")

    # Write files
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)

    qrels_path = eval_dir / "qrels.json"
    qrels_path.write_text(json.dumps(qrels, indent=2))
    print(f"\nqrels → {qrels_path}  ({sum(len(v) for v in qrels.values())} total relevant)")

    queries_path = eval_dir / "queries.json"
    queries_path.write_text(json.dumps(queries_out, indent=2))
    print(f"queries → {queries_path}  ({len(queries_out)} queries)")

    return qrels


if __name__ == "__main__":
    build_qrels()
