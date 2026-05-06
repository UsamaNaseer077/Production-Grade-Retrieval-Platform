# Failure Modes — Honest Documentation

This document describes known limitations and failure modes of the retrieval platform.
Every section includes the observed behaviour, root cause, and mitigation strategies.

---

## 1. Retrieval Failures

### 1.1 BM25 Vocabulary Mismatch
**Symptom**: Semantically relevant documents rank low or are absent in BM25 results.  
**Root cause**: BM25 relies on exact token overlap. A query like "how does gradient optimisation work"
will not match chunks that say "gradient descent updates weights", even though they are the same
concept.  
**Mitigation**: Hybrid search (BM25 + dense) covers this. Dense retrieval catches paraphrases
and synonyms; RRF fusion combines both signals.  
**When it still hurts**: Highly domain-specific acronyms not seen during embedding training
(e.g., internal product codenames) may not be captured by either method.

### 1.2 Dense Retrieval Semantic Drift
**Symptom**: Top-k dense results are semantically similar but topically wrong.  
**Root cause**: The `all-MiniLM-L6-v2` bi-encoder is a general-purpose model. On narrow technical
domains, it may conflate similar-sounding but distinct concepts (e.g., "transformer" the
neural architecture vs "transformer" the electrical component).  
**Mitigation**: Fine-tune the embedding model on domain data using contrastive learning.
Use BEIR benchmarks to select the best base model for your domain.  
**Observed example**: Queries about "attention mechanism in ML" occasionally retrieve electrical
engineering content about "attention signals".

### 1.3 Cross-Encoder Score Range
**Symptom**: `final_score` values in search results are negative (e.g., -10.5).  
**Root cause**: The ms-marco-MiniLM cross-encoder outputs raw logits (unbounded), not
probabilities. Negative scores are expected and simply mean lower relevance than positive scores.  
**Not a bug**: The ranking order is correct; only the magnitude is unintuitive.  
**Display fix**: Normalise with sigmoid if you need [0, 1] scores for display.

### 1.4 Recall@k vs MRR Divergence
**Observed**: MRR ≈ 0.74 but recall@1 ≈ 0.04 on the eval set.  
**Why**: The qrels contain 10–19 relevant chunks per query (all paragraphs from topic files).
recall@k = hits_in_top_k / total_relevant. With 15 relevant chunks and top-k=10,
maximum achievable recall@10 ≈ 0.67. The high MRR means the *first* relevant result
appears at rank ~1.4 on average — retrieval precision is good.  
**Implication**: For RAG where only top-1 context is needed, the system performs well.
For exhaustive recall (find all relevant passages), top-k should be increased.

---

## 2. Ingestion Failures

### 2.1 Incremental Hash Inconsistency
**Symptom**: `hashes.json` contains file entries but the FAISS index lacks their chunks.  
**Root cause**: The hash store is updated in `IngestPipeline.ingest_file()` at parse time,
before `VectorStore.add_chunks()` is called. If the process is killed between parse and index,
the hash store marks the file "seen" but the vectors are never written.  
**Fix**: Run `python scripts/rebuild_index.py --dirs <dir>` which detects and repairs the
inconsistency by removing stale hashes and re-indexing.  
**Prevention**: A transactional write (index + hash store together) would eliminate this.

### 2.2 BM25 Full Rebuild on Every Add
**Symptom**: BM25 indexing is slow when adding new batches to a large existing index.  
**Root cause**: `BM25Okapi` in rank-bm25 does not support incremental updates. Every call to
`add_chunks()` rebuilds the full inverted index over all documents.  
**Impact**: At 100K chunks, BM25 rebuild takes ~30 seconds. At 1M chunks, this becomes
impractical.  
**Mitigation**: For >100K documents, replace `BM25Okapi` with Elasticsearch, OpenSearch, or
Typesense which support streaming index updates without full rebuilds.

### 2.3 Short or Empty Documents
**Symptom**: Some files produce 0 chunks despite being ingested (hash recorded, 0 row-docs).  
**Root cause**: Text normalisation (`_norm`) strips excessive whitespace. Files that consist
mostly of whitespace, binary content, or HTML navigation menus (e.g., scraped web pages)
produce near-empty text that fails the `if not text:` guard.  
**Observed**: Shakespeare files downloaded via `download_shakespeare.py` contain website
navigation HTML rather than play text — producing minimal extractable content.  
**Fix**: Validate ingested character count in the hash store and warn on files < 100 characters
after normalisation.

### 2.4 PDF Extraction Quality
**Symptom**: PDF chunks contain garbled text, merged words, or missing structure.  
**Root cause**: PDFs without embedded text (scanned, image-only) require OCR. `pypdf` extracts
structural text but misses tables and multi-column layouts. `docling` handles these cases better
but requires additional system dependencies.  
**Fix for scanned PDFs**: Install `docling` + `tesseract` (`brew install tesseract`).  
**Fix for table-heavy PDFs**: Use `docling` with `do_table_structure=True` (the default).

---

## 3. LLM and Generation Failures

### 3.1 LLM Offline (No Backend)
**Symptom**: Answers return `[LLM offline — showing top retrieved passage]`.  
**Root cause**: No LLM backend is configured or reachable.  
**Fix**:
```bash
# Option A — Local (free, private)
ollama serve && ollama pull llama3.2

# Option B — Cloud (paid)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3.2 Faithfulness Self-Check False Positives
**Symptom**: Self-checker reports `verdict: warn` on factually correct answers.  
**Root cause**: Token-overlap faithfulness is a heuristic proxy. Short claims (< 15 tokens)
are skipped. Claims that paraphrase rather than reuse context tokens score low even if grounded.  
**Reliability**: ~80% precision on clearly grounded/ungrounded pairs. Not suitable as a
production safety gate without a proper entailment model.  
**Upgrade**: Replace with [MiniCheck](https://github.com/Liyan06/MiniCheck) or Vectara's
[Hallucination Evaluation Model (HEM)](https://huggingface.co/vectara/hallucination_evaluation_model)
for production faithfulness scoring.

### 3.3 Context Window Overflow
**Symptom**: LLM truncates or ignores later context chunks.  
**Root cause**: `top_k=10` × `parent_size=1024` tokens = ~10,000 tokens of context, which
exceeds the effective context window of smaller models (TinyLlama: 2048 tokens).  
**Fix**: Reduce `top_k_rerank` to 3–5 for small models, or use a model with 32K+ context.

---

## 4. Scale Failures

### 4.1 HNSW Memory Usage
**Symptom**: Memory usage grows proportionally with index size; OOM at scale.  
**Root cause**: FAISS HNSW stores all vectors in RAM. At 384 dimensions (float32), each
vector is 1.5 KB. At 1M vectors: ~1.5 GB just for vectors, plus HNSW graph edges (~32 M edges
for M=32 → additional ~256 MB).  
**Capacity estimates**:
- 16 GB RAM → ~8M vectors at 384d (safe)  
- 1B vectors → requires DiskANN, SPANN, or quantisation (SQ8 reduces to 384 bytes/vec)  
**Production path**: Replace `VectorStore` with Qdrant (exact same retrieval interface,
adds filtering, sharding, and on-disk HNSW).

### 4.2 Pickle Serialisation Security
**Symptom**: N/A — latent risk.  
**Root cause**: `bm25.pkl` uses Python `pickle` for serialisation. Loading pickles from
untrusted sources enables arbitrary code execution.  
**Fix**: Only load BM25 pickles from trusted, write-protected storage. For multi-tenant or
user-uploaded data scenarios, serialise to JSON or use a proper search engine.

### 4.3 Single-Process GIL Contention
**Symptom**: CPU-bound embedding throughput does not scale with worker count.  
**Root cause**: Python's GIL serialises CPU-bound threads. `uvicorn --workers N` uses
multiple processes (bypasses GIL) but each process holds its own copy of the index.  
**Fix**: Serve the FAISS index via a dedicated gRPC microservice; use async FastAPI workers
only for HTTP orchestration.

---

## 5. Evaluation Limitations

### 5.1 Auto-Seeded Qrels Are Not Gold Labels
**Symptom**: Eval metrics look good but may be optimistic.  
**Root cause**: `build_qrels.py` identifies relevant chunks by source-pattern + keyword matching,
not human judgement. The model may have indexed content it shouldn't retrieve, and some
relevant chunks may be missed.  
**Fix**: For production evaluation, use human annotators or LLM-as-judge to label a random
sample of query-chunk pairs, then extrapolate with pooling.

### 5.2 Evaluation Latency Includes Cross-Encoder
**Observed**: p50 latency ~1.07s per query.  
**Breakdown**:
- Dense search (HNSW): ~5–20ms  
- BM25 search: ~5–30ms  
- Cross-encoder (10×50 pairs): ~800–1000ms on CPU (MPS on Apple Silicon: ~200ms)  
**Fix for low-latency production**: Use cross-encoder only as an asynchronous quality signal,
or replace with ColBERT late-interaction for sub-100ms reranking at scale.

---

## 6. What Works Well

The following aspects are functioning correctly and are suitable for production use:

| Feature | Status | Notes |
|---|---|---|
| Incremental ingestion | ✅ | SHA-256 content hash; O(1) skip check |
| Embedding disk cache | ✅ | Warm restart ≈ 0ms re-embed cost |
| BM25 + dense hybrid | ✅ | RRF fusion; always outperforms either alone |
| Cross-encoder reranking | ✅ | Significant precision improvement |
| Parent-child chunking | ✅ | Best context quality for RAG |
| Latency tracking | ✅ | p50/p95/p99 per component |
| Self-check faithfulness | ⚠️ | Heuristic proxy only; 80% precision |
| BM25 incremental updates | ❌ | Full rebuild on every add |
| LLM answer generation | ⚠️ | Requires separate LLM backend |
| Billion-scale vectors | ❌ | FAISS HNSW limited to ~8M on 16GB RAM |
