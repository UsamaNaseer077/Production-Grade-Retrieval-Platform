"""
FastAPI application — Production Retrieval Platform

Endpoints
---------
POST /api/v1/ingest/file          ingest a single file (by path on server)
POST /api/v1/ingest/upload        upload + ingest a file directly
POST /api/v1/ingest/dir           ingest an entire directory
GET  /api/v1/ingest/stats         index stats

POST /api/v1/query                full RAG query (retrieve + LLM + self-check)
POST /api/v1/query/retrieve       retrieval only (no LLM)

POST /api/v1/eval/run             run evaluation harness
GET  /api/v1/eval/report          return last saved report

GET  /api/v1/admin/health         service health
GET  /api/v1/admin/stats          latency percentiles per component
POST /api/v1/admin/reload-prompt  hot-reload system.md
GET  /api/v1/admin/models         list available Ollama models

GET  /health                      simple liveness probe (for Nginx / k8s)
"""
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import CFG
from src.embeddings import CachedEmbedder
from src.evaluation.harness import print_report, run_eval
from src.generation import LLMWrapper, PromptManager, SelfChecker
from src.indexing.bm25_store import BM25Store
from src.indexing.embedder import build_embedder
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.pipeline import HashStore, IngestPipeline
from src.memory import LongTermMemory, ShortTermMemory
from src.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from src.telemetry import TRACKER

logger = logging.getLogger(__name__)


# ── app-wide state ────────────────────────────────────────────────────────────

class AppState:
    embedder: Any = None
    vs: VectorStore = None
    bm25: BM25Store = None
    retriever: HybridRetriever = None
    chunker: Chunker = None
    pipeline: IngestPipeline = None
    hash_store: HashStore = None
    llm: LLMWrapper = None
    stm: ShortTermMemory = None
    ltm: LongTermMemory = None
    pm: PromptManager = None
    checker: SelfChecker = None
    last_eval_report: dict = {}


STATE = AppState()


# ── lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    logger.info("Starting up retrieval platform …")

    STATE.embedder = build_embedder(
        model_name=CFG.embedding.model_name,
        lsa_dim=CFG.embedding.lsa_dim,
        lsa_path=CFG.index.lsa_path,
        cache_dir=CFG.embedding.cache_dir,
        use_cache=True,
    )
    STATE.vs = VectorStore(
        lsa_dim=CFG.embedding.lsa_dim,
        hnsw_m=CFG.index.hnsw_m,
        index_path=CFG.index.index_path,
        meta_path=CFG.index.meta_path,
        lsa_path=CFG.index.lsa_path,
        embedder=STATE.embedder,
    )
    STATE.bm25 = BM25Store(CFG.index.bm25_path)
    reranker = CrossEncoderReranker()
    STATE.retriever = HybridRetriever(
        STATE.vs, STATE.bm25, reranker,
        top_k_dense=CFG.retrieval.top_k_dense,
        top_k_sparse=CFG.retrieval.top_k_sparse,
        top_k_rerank=CFG.retrieval.top_k_rerank,
        rrf_k=CFG.retrieval.rrf_k,
    )
    STATE.chunker = Chunker(
        strategy=CFG.chunking.strategy,
        child_size=CFG.chunking.child_size,
        parent_size=CFG.chunking.parent_size,
        overlap=CFG.chunking.overlap,
    )
    STATE.hash_store = HashStore(CFG.index.hash_path)
    STATE.pipeline = IngestPipeline(STATE.hash_store)

    STATE.llm = LLMWrapper(
        model_name=CFG.llm.model_name,
        backend=CFG.llm.backend,
        temperature=CFG.llm.temperature,
        max_tokens=CFG.llm.max_tokens,
        timeout=CFG.llm.timeout,
    )
    STATE.stm = ShortTermMemory(
        max_turns=CFG.memory.stm_max_turns,
        ttl_seconds=CFG.memory.stm_ttl_seconds,
    )
    STATE.ltm = LongTermMemory(db_path=str(CFG.memory.ltm_db_path))
    STATE.pm = PromptManager(prompt_path=str(CFG.llm.prompt_path))
    STATE.checker = SelfChecker()

    logger.info(
        "Ready — index size: %d vectors | LLM: %s",
        STATE.vs.size,
        "online" if STATE.llm.available else "offline",
    )

    yield  # ── app runs ──

    # ── shutdown ─────────────────────────────────────────────────────────────
    logger.info("Saving indexes …")
    STATE.vs.save()
    STATE.bm25.save()
    STATE.hash_store.save()
    logger.info("Shutdown complete.")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Retrieval Platform",
    description="Production-grade hybrid search + RAG with memory and self-check",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────

class IngestFileRequest(BaseModel):
    path: str
    strategy: Optional[str] = None  # override chunk strategy

class IngestDirRequest(BaseModel):
    directory: str
    recursive: bool = True
    strategy: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = 10
    use_llm: bool = True
    return_sources: bool = True

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_ingest(docs, strategy: Optional[str] = None) -> dict:
    if not docs:
        return {"new_docs": 0, "new_chunks": 0}
    chunker = (
        Chunker(strategy=strategy, child_size=CFG.chunking.child_size,
                parent_size=CFG.chunking.parent_size, overlap=CFG.chunking.overlap)
        if strategy else STATE.chunker
    )
    chunks = chunker.chunk_many(docs)
    da = STATE.vs.add_chunks(chunks)
    sa = STATE.bm25.add_chunks(chunks)
    STATE.vs.save()
    STATE.bm25.save()
    STATE.hash_store.save()
    return {"new_docs": len(docs), "new_chunks": da, "bm25_added": sa}


def _results_to_dicts(results, top_k: int) -> List[dict]:
    out = []
    for r in results[:top_k]:
        out.append({
            "rank": len(out) + 1,
            "chunk_id": r.chunk_id,
            "source": r.source,
            "text": r.text,
            "parent_text": r.parent_text,
            "rrf_score": round(r.rrf_score, 5),
            "final_score": round(r.final_score, 5),
            "reranked": r.reranked,
            "dense_rank": r.dense_rank,
            "sparse_rank": r.sparse_rank,
        })
    return out


# ── routes: ingest ────────────────────────────────────────────────────────────

@app.post("/api/v1/ingest/file")
def ingest_file(req: IngestFileRequest):
    t0 = TRACKER.start()
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(404, f"File not found: {path}")
    docs = STATE.pipeline.ingest_file(path)
    result = _run_ingest(docs, req.strategy)
    result["latency_ms"] = TRACKER.elapsed_ms(t0)
    return result


@app.post("/api/v1/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    strategy: Optional[str] = None,
):
    t0 = TRACKER.start()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        docs = STATE.pipeline.ingest_file(tmp_path)
        # fix metadata filename to original
        for d in docs:
            d.metadata["filename"] = file.filename
            d.source = file.filename
        result = _run_ingest(docs, strategy)
    finally:
        tmp_path.unlink(missing_ok=True)
    result["filename"] = file.filename
    result["latency_ms"] = TRACKER.elapsed_ms(t0)
    return result


@app.post("/api/v1/ingest/dir")
def ingest_dir(req: IngestDirRequest):
    t0 = TRACKER.start()
    d = Path(req.directory)
    if not d.exists():
        raise HTTPException(404, f"Directory not found: {d}")
    docs = STATE.pipeline.ingest_dir(d, recursive=req.recursive)
    result = _run_ingest(docs, req.strategy)
    result["latency_ms"] = TRACKER.elapsed_ms(t0)
    return result


@app.get("/api/v1/ingest/stats")
def ingest_stats():
    return {
        "vector_index_size": STATE.vs.size,
        "bm25_index_size": STATE.bm25.size,
        "ltm_records": STATE.ltm.count(),
        "stm_active_sessions": STATE.stm.active_sessions,
        "embedding_cache": (
            STATE.embedder.cache_stats
            if hasattr(STATE.embedder, "cache_stats")
            else {}
        ),
    }


# ── routes: query ─────────────────────────────────────────────────────────────

@app.post("/api/v1/query")
def query(req: QueryRequest):
    if STATE.vs.size == 0:
        raise HTTPException(400, "Index is empty — run /api/v1/ingest first")

    t_total = TRACKER.start()

    # 1. Retrieve
    with TRACKER.track("dense_search"):
        pass  # timing is inside HybridRetriever; we track total retrieval below
    t_ret = TRACKER.start()
    results = STATE.retriever.search(req.query)
    ret_ms = TRACKER.elapsed_ms(t_ret)
    TRACKER.record("retrieval", ret_ms)

    contexts = _results_to_dicts(results, req.top_k)

    # 2. Memory context injection
    conv_history = STATE.stm.format_for_prompt(req.session_id)
    mem_context = STATE.ltm.format_relevant(req.query, session_id=req.session_id)

    answer_text = ""
    llm_result = {"latency_ms": 0.0, "model": "none"}
    self_check_result = {}

    if req.use_llm:
        # 3. Build prompt
        user_prompt = STATE.pm.build(
            query=req.query,
            contexts=contexts,
            conversation_history=conv_history,
            memory_context=mem_context,
        )

        # 4. Generate
        t_llm = TRACKER.start()
        llm_result = STATE.llm.generate(STATE.pm.system_prompt, user_prompt)
        TRACKER.record("llm", TRACKER.elapsed_ms(t_llm))
        answer_text = llm_result["text"]

        # 5. Self-check
        t_sc = TRACKER.start()
        self_check_result = STATE.checker.check(answer_text, contexts)
        TRACKER.record("self_check", TRACKER.elapsed_ms(t_sc))

        # 5b. Re-generate if faithfulness failed
        if self_check_result.get("need_regen"):
            user_prompt_strict = STATE.pm.build(
                query=req.query, contexts=contexts,
                conversation_history=conv_history, memory_context=mem_context,
                grounding_reminder=True,
            )
            llm_result2 = STATE.llm.generate(STATE.pm.system_prompt, user_prompt_strict)
            answer_text = llm_result2["text"]
            self_check_result = STATE.checker.check(answer_text, contexts)
            self_check_result["regenerated"] = True

        # 6. Memory write-back
        STATE.stm.add(req.session_id, "user", req.query)
        STATE.stm.add(req.session_id, "assistant", answer_text)
        STATE.ltm.store(
            session_id=req.session_id,
            question=req.query,
            answer=answer_text,
            source_ids=[c["chunk_id"] for c in contexts],
        )
    else:
        answer_text = contexts[0]["parent_text"] if contexts else "No results found."

    total_ms = TRACKER.elapsed_ms(t_total)
    TRACKER.record_request(total_ms)

    return {
        "query": req.query,
        "session_id": req.session_id,
        "answer": answer_text,
        "sources": contexts if req.return_sources else [],
        "self_check": self_check_result,
        "latency_breakdown": {
            "retrieval_ms": round(ret_ms, 1),
            "llm_ms": llm_result.get("latency_ms", 0),
            "total_ms": round(total_ms, 1),
            "llm_model": llm_result.get("model", "none"),
        },
    }


@app.post("/api/v1/query/retrieve")
def retrieve_only(req: RetrieveRequest):
    if STATE.vs.size == 0:
        raise HTTPException(400, "Index is empty")
    t0 = TRACKER.start()
    results = STATE.retriever.search(req.query)
    return {
        "query": req.query,
        "results": _results_to_dicts(results, req.top_k),
        "latency_ms": TRACKER.elapsed_ms(t0),
    }


# ── routes: eval ──────────────────────────────────────────────────────────────

@app.post("/api/v1/eval/run")
def eval_run(
    queries_path: str = str(CFG.ev.queries_path),
    qrels_path: str = str(CFG.ev.qrels_path),
):
    if not Path(queries_path).exists():
        raise HTTPException(404, f"queries file not found: {queries_path}")
    if not Path(qrels_path).exists():
        raise HTTPException(404, f"qrels file not found: {qrels_path}")
    report = run_eval(
        STATE.retriever, queries_path, qrels_path, CFG.ev.k_values
    )
    STATE.last_eval_report = report
    # persist
    out = Path("eval/report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(__import__("json").dumps(report, indent=2))
    return report


@app.get("/api/v1/eval/report")
def eval_report():
    p = Path("eval/report.json")
    if p.exists():
        return __import__("json").loads(p.read_text())
    return STATE.last_eval_report or {"detail": "No report yet — run /api/v1/eval/run"}


# ── routes: admin ─────────────────────────────────────────────────────────────

@app.get("/api/v1/admin/health")
def admin_health():
    return {
        "status": "ok",
        "vector_index_size": STATE.vs.size,
        "bm25_index_size": STATE.bm25.size,
        "llm_available": STATE.llm.available,
        "llm_model": STATE.llm.model_name,
        "ltm_records": STATE.ltm.count(),
    }


@app.get("/api/v1/admin/stats")
def admin_stats():
    return TRACKER.all_stats()


@app.post("/api/v1/admin/reload-prompt")
def reload_prompt():
    STATE.pm.reload()
    return {"status": "ok", "prompt_preview": STATE.pm.system_prompt[:200]}


@app.get("/api/v1/admin/models")
def list_models():
    return {"models": STATE.llm.list_models()}


# ── liveness probe ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
