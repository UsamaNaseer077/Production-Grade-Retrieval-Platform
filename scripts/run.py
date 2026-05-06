#!/usr/bin/env python3
"""
CLI:  python scripts/run.py <cmd> [options]

  generate-data              create sample corpus + eval stubs
  ingest  --dir PATH         ingest documents (incremental, all formats)
  search  --query TEXT       hybrid search + optional LLM answer
  eval                       run evaluation harness (populate qrels first)
  serve                      start FastAPI server
  memory  --session SID      show long-term memory for a session
  cache-stats                print embedding cache hit/miss stats
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CFG
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import build_embedder
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.pipeline import HashStore, IngestPipeline
from src.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from src.evaluation.harness import run_eval, print_report
from src.generation import LLMWrapper, PromptManager, SelfChecker
from src.memory import ShortTermMemory, LongTermMemory
from src.telemetry import TRACKER

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── shared factory ────────────────────────────────────────────────────────────

def _build_retriever():
    emb = build_embedder(
        model_name=CFG.embedding.model_name,
        lsa_dim=CFG.embedding.lsa_dim,
        lsa_path=CFG.index.lsa_path,
        cache_dir=CFG.embedding.cache_dir,
        use_cache=True,
    )
    vs  = VectorStore(
        lsa_dim=CFG.embedding.lsa_dim, hnsw_m=CFG.index.hnsw_m,
        index_path=CFG.index.index_path, meta_path=CFG.index.meta_path,
        lsa_path=CFG.index.lsa_path, embedder=emb,
    )
    bm  = BM25Store(CFG.index.bm25_path)
    rr  = CrossEncoderReranker()
    return HybridRetriever(
        vs, bm, rr,
        top_k_dense=CFG.retrieval.top_k_dense,
        top_k_sparse=CFG.retrieval.top_k_sparse,
        top_k_rerank=CFG.retrieval.top_k_rerank,
        rrf_k=CFG.retrieval.rrf_k,
    ), vs, bm, emb


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_data(_):
    import random
    rng = random.Random(42)
    d = Path("data/sample_docs")
    d.mkdir(parents=True, exist_ok=True)

    topics = {
        "retrieval": [
            "BM25 is a sparse retrieval method using term and document frequency.",
            "Dense retrieval encodes text as vectors and uses approximate nearest neighbour search.",
            "Hybrid retrieval combines sparse BM25 and dense embeddings for better coverage.",
            "Reciprocal Rank Fusion merges ranked lists without requiring score normalisation.",
            "A cross-encoder re-ranker scores query and document jointly for higher precision.",
        ],
        "machine_learning": [
            "Gradient descent minimises a loss function by iteratively updating model weights.",
            "Overfitting is when a model memorises training data but fails on unseen examples.",
            "Transfer learning reuses pre-trained model weights for a new downstream task.",
            "Regularisation penalises large weights to reduce overfitting and improve generalisation.",
            "The bias-variance trade-off describes the tension between model complexity and fit.",
        ],
        "software_engineering": [
            "Continuous integration tests code changes automatically on every commit.",
            "Docker packages applications with dependencies for reproducible deployments.",
            "Observability in distributed systems relies on metrics, logs, and traces.",
            "Database indexing creates auxiliary structures to accelerate query performance.",
            "API versioning allows backward-compatible changes while supporting older clients.",
        ],
    }

    for name, sents in topics.items():
        (d / f"{name}.txt").write_text(
            "\n\n".join(s + "  " + " ".join(rng.choices(sents, k=2)) for s in sents * 5)
        )

    csv_p = d / "glossary.csv"
    with csv_p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "definition", "domain"])
        for domain, sents in topics.items():
            for s in sents:
                w.writerow([s.split()[0], s, domain])

    ev = Path("eval")
    ev.mkdir(exist_ok=True)
    queries = [
        {"query_id": "q1", "text": "how does BM25 sparse retrieval work"},
        {"query_id": "q2", "text": "what is hybrid retrieval"},
        {"query_id": "q3", "text": "explain gradient descent"},
        {"query_id": "q4", "text": "what is overfitting in machine learning"},
        {"query_id": "q5", "text": "how does containerisation work with Docker"},
    ]
    (ev / "queries.json").write_text(json.dumps(queries, indent=2))
    (ev / "qrels.json").write_text(json.dumps({q["query_id"]: [] for q in queries}, indent=2))

    print("Sample data ready.")
    print("Next: python scripts/run.py ingest --dir data/sample_docs")
    print("NOTE: eval/qrels.json has empty stubs — populate after first ingest.")


def cmd_ingest(args):
    d = Path(args.dir)
    if not d.exists():
        print(f"Not found: {d}")
        return

    hs  = HashStore(CFG.index.hash_path)
    pip = IngestPipeline(hs)
    ch  = Chunker(
        args.strategy,
        CFG.chunking.child_size,
        CFG.chunking.parent_size,
        CFG.chunking.overlap,
    )

    with TRACKER.track("ingest_total"):
        docs = pip.ingest_dir(d, not args.flat)

    if not docs:
        print("No new documents — everything up to date.")
        return

    chunks = ch.chunk_many(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} docs")

    emb = build_embedder(
        model_name=CFG.embedding.model_name,
        lsa_dim=CFG.embedding.lsa_dim,
        lsa_path=CFG.index.lsa_path,
        cache_dir=CFG.embedding.cache_dir,
        use_cache=True,
    )
    vs  = VectorStore(
        lsa_dim=CFG.embedding.lsa_dim, hnsw_m=CFG.index.hnsw_m,
        index_path=CFG.index.index_path, meta_path=CFG.index.meta_path,
        lsa_path=CFG.index.lsa_path, embedder=emb,
    )
    bm  = BM25Store(CFG.index.bm25_path)

    with TRACKER.track("embed_and_index"):
        da = vs.add_chunks(chunks)
        sa = bm.add_chunks(chunks)

    vs.save()
    bm.save()

    print(f"Dense +{da}  BM25 +{sa}  total_vectors={vs.size}")
    if hasattr(emb, "cache_stats"):
        print(f"Embedding cache: {emb.cache_stats}")
    print(f"Ingest latency: {TRACKER.stats('ingest_total')['mean_ms']:.0f} ms")


def cmd_search(args):
    retriever, vs, bm, emb = _build_retriever()
    llm  = LLMWrapper(
        model_name=CFG.llm.model_name,
        backend=CFG.llm.backend,
        temperature=CFG.llm.temperature,
        max_tokens=CFG.llm.max_tokens,
    )
    pm      = PromptManager(str(CFG.llm.prompt_path))
    checker = SelfChecker()

    with TRACKER.track("total"):
        with TRACKER.track("retrieval"):
            results = retriever.search(args.query)

    print(f"\n── Retrieved {len(results)} results ──")
    contexts = []
    for i, r in enumerate(results):
        snippet = (r.parent_text[:300] + "…") if len(r.parent_text) > 300 else r.parent_text
        score_label = "ce" if r.reranked else "rrf"
        print(f"\n[{i+1}] {score_label}={r.final_score:.4f}  rrf={r.rrf_score:.4f}  dense#{r.dense_rank}  sparse#{r.sparse_rank}")
        print(f"    {r.source}")
        print(f"    {snippet}")
        contexts.append({
            "source": r.source, "text": r.text, "parent_text": r.parent_text,
            "chunk_id": r.chunk_id,
        })

    if args.llm and llm.available:
        user_prompt = pm.build(query=args.query, contexts=contexts)
        print("\n── Generating answer … ──")
        with TRACKER.track("llm"):
            llm_result = llm.generate(pm.system_prompt, user_prompt)
        answer = llm_result["text"]
        check  = checker.check(answer, contexts)
        print(f"\nAnswer ({llm_result['model']}):")
        print(answer)
        print(f"\nFaithfulness: {check['verdict']}  score={check['score']}  "
              f"grounded={check['grounded']}")
        if check["issues"]:
            print("Potential issues:")
            for iss in check["issues"]:
                print(f"  • {iss}")
    elif args.llm:
        print("\n[LLM offline — start Ollama and pull a model]")

    # latency breakdown
    stats = TRACKER.all_stats()
    print("\n── Latency ──")
    for comp, s in stats.items():
        if s["count"]:
            print(f"  {comp:<15} p50={s['p50_ms']} ms")


def cmd_eval(_):
    retriever, _, _, _ = _build_retriever()
    rep = run_eval(
        retriever,
        CFG.ev.queries_path,
        CFG.ev.qrels_path,
        CFG.ev.k_values,
    )
    print_report(rep)
    Path("eval/report.json").write_text(json.dumps(rep, indent=2))
    print("\nFull report → eval/report.json")


def cmd_serve(args):
    import uvicorn
    print(f"Starting API on http://0.0.0.0:{args.port}")
    print("Docs: http://localhost:{args.port}/docs")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
        log_level="info",
        reload=args.reload,
    )


def cmd_memory(args):
    ltm = LongTermMemory(str(CFG.memory.ltm_db_path))
    print(f"LTM total records: {ltm.count()}")
    if args.session:
        history = ltm.get_session_history(args.session, limit=20)
        if not history:
            print(f"No records for session '{args.session}'")
            return
        for r in history:
            from datetime import datetime
            ts = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M")
            print(f"\n[{ts}] Q: {r.question}")
            print(f"       A: {r.answer[:200]}…" if len(r.answer) > 200 else f"       A: {r.answer}")


def cmd_cache_stats(_):
    emb = build_embedder(
        model_name=CFG.embedding.model_name,
        lsa_dim=CFG.embedding.lsa_dim,
        lsa_path=CFG.index.lsa_path,
        cache_dir=CFG.embedding.cache_dir,
        use_cache=True,
    )
    if hasattr(emb, "cache_stats"):
        stats = emb.cache_stats
        print(f"Hits   : {stats['hits']}")
        print(f"Misses : {stats['misses']}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        print(f"Cached files: {stats['cached_files']}")
    else:
        print("Embedding cache not enabled")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Retrieval Platform CLI")
    s = p.add_subparsers(dest="cmd", required=True)

    s.add_parser("generate-data", help="Create sample corpus + eval stubs")

    pi = s.add_parser("ingest", help="Ingest documents (incremental)")
    pi.add_argument("--dir", default="data/sample_docs")
    pi.add_argument("--strategy", default="parent_child",
                    choices=["fixed", "sentence", "parent_child", "sliding", "structural"])
    pi.add_argument("--flat", action="store_true", help="Non-recursive directory scan")

    ps = s.add_parser("search", help="Search + optional LLM answer")
    ps.add_argument("--query", required=True)
    ps.add_argument("--llm", action="store_true", help="Generate answer with LLM")

    s.add_parser("eval", help="Run evaluation harness")

    pserve = s.add_parser("serve", help="Start FastAPI server")
    pserve.add_argument("--port", type=int, default=8000)
    pserve.add_argument("--workers", type=int, default=1)
    pserve.add_argument("--reload", action="store_true")

    pm = s.add_parser("memory", help="Inspect long-term memory")
    pm.add_argument("--session", default=None)

    s.add_parser("cache-stats", help="Show embedding cache statistics")

    args = p.parse_args()
    {
        "generate-data": cmd_data,
        "ingest": cmd_ingest,
        "search": cmd_search,
        "eval": cmd_eval,
        "serve": cmd_serve,
        "memory": cmd_memory,
        "cache-stats": cmd_cache_stats,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
