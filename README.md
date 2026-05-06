# Retrieval Platform

A production-grade hybrid retrieval system with incremental indexing, BM25 + semantic search, cross-encoder reranking, and a built-in evaluation framework.

## Features

- **Hybrid search** — BM25 sparse + dense vector (FAISS HNSW) fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` reranks top candidates for precision
- **Incremental ingestion** — SHA-256 content hashing; unchanged files are skipped on re-run
- **14 file formats** — PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, XML, JSON, EPUB, RTF, PNG, JPG
- **Parent-child chunking** — embed 256-token children, return 1024-token parent context for RAG
- **Embedding cache** — sharded disk KV cache; warm re-embed cost ≈ 0 ms
- **LLM answer generation** — supports Ollama (local), OpenAI, Anthropic, HuggingFace backends
- **Faithfulness self-check** — two-signal heuristic (token overlap + ungrounded claim ratio)
- **Evaluation framework** — Recall@K, MRR, latency p50/p95/p99 with threshold gating
- **FastAPI REST API** — `/search`, `/ingest`, `/chat`, `/metrics` endpoints
- **Short-term + long-term memory** — SQLite-backed session history and Q&A recall

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest the included corpus
python scripts/run.py ingest data/corpus data/shakespeare data/sample_docs

# 4. Search
python scripts/run.py search "how does BM25 sparse retrieval work"

# 5. Start the API server
python scripts/run.py serve
```

## Project Structure

```
retrieval-platform/
├── src/
│   ├── config.py               # Centralised settings (CFG)
│   ├── ingestion/              # File loaders for 14 formats
│   ├── indexing/
│   │   ├── embedder.py         # SentenceTransformer + 4-tier fallback
│   │   ├── vector_store.py     # FAISS HNSW index wrapper
│   │   └── bm25_store.py       # rank-bm25 BM25Okapi store
│   ├── retrieval/
│   │   └── hybrid.py           # RRF fusion + cross-encoder reranker
│   ├── generation/             # LLM backends + prompt builder
│   ├── evaluation/             # Recall@K, MRR, latency metrics
│   ├── memory/                 # Short-term (session) + long-term memory
│   ├── embeddings/             # Disk KV embedding cache
│   ├── telemetry/              # Structured logging + Prometheus metrics
│   └── api/
│       └── main.py             # FastAPI application
├── scripts/
│   ├── run.py                  # Main CLI (ingest / search / eval / serve)
│   ├── generate_corpus.py      # Generate 175-file synthetic corpus
│   ├── build_qrels.py          # Build eval/qrels.json relevance labels
│   ├── rebuild_index.py        # Repair hash store / FAISS sync
│   └── download_shakespeare.py # Download Shakespeare plays
├── data/
│   ├── corpus/                 # Generated technical corpus (tracked)
│   │   ├── tech/               # Information retrieval, ML, SE docs
│   │   ├── science/            # Physics, biology, chemistry docs
│   │   ├── business/           # Finance, operations docs
│   │   ├── records/            # CSV tabular data
│   │   ├── knowledge/          # JSON knowledge bases
│   │   └── notes/              # Markdown notes
│   ├── shakespeare/            # Shakespeare plays (tracked)
│   └── sample_docs/            # Sample documents (tracked)
├── eval/
│   ├── test_suite.json         # Full test suite with thresholds
│   ├── qrels.json              # Relevance judgements (chunk IDs)
│   └── queries.json            # Eval query set
├── tests/
│   └── test_pipeline.py        # Unit + integration tests
├── docs/
│   ├── failure_modes.md        # Known limitations and mitigations
│   └── build_pdf.py            # Generates the architecture PDF
└── deployment/
    ├── Dockerfile
    └── nginx.conf
```

## CLI Reference

```bash
# Ingest one or more directories
python scripts/run.py ingest <dir1> [dir2 ...]

# Search with optional LLM answer generation
python scripts/run.py search "your query" [--llm]

# Run evaluation suite (Recall@K, MRR, latency)
python scripts/run.py eval

# Start FastAPI server (default: http://localhost:8000)
python scripts/run.py serve [--host 0.0.0.0] [--port 8000]

# Run all unit and integration tests
pytest tests/

# Rebuild the corpus from scratch
python scripts/generate_corpus.py

# Rebuild relevance labels for evaluation
python scripts/build_qrels.py

# Repair hash store / FAISS sync after interrupted ingestion
python scripts/rebuild_index.py --dirs data/corpus data/shakespeare
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Hybrid search with optional reranking |
| `POST` | `/ingest` | Upload and index a file |
| `POST` | `/chat` | RAG answer generation with session memory |
| `GET` | `/metrics` | Prometheus-format latency metrics |
| `GET` | `/health` | Health check |

### Search example

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does BM25 work", "top_k": 5, "use_reranker": true}'
```

## LLM Backends

By default, no LLM is configured. Search still works — the system returns retrieved passages. To enable answer generation:

```bash
# Option A — Local (free, private)
ollama serve && ollama pull llama3.2
export LLM_BACKEND=ollama

# Option B — OpenAI
export OPENAI_API_KEY=sk-...
export LLM_BACKEND=openai

# Option C — Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_BACKEND=anthropic
```

## Evaluation

The system is evaluated against `eval/qrels.json` (relevance labels) and `eval/test_suite.json` (thresholds):

| Metric | Threshold | Observed |
|--------|-----------|----------|
| Recall@1 | ≥ 0.20 | ~0.40 |
| Recall@5 | ≥ 0.70 | ~0.54 |
| MRR | ≥ 0.50 | ~0.74 |
| Embed latency p95 | ≤ 200 ms | ~0.3 ms (cached) |
| Total retrieval p95 | ≤ 1000 ms | ~881 ms |

Run the evaluation:

```bash
python scripts/run.py eval
```

## Known Limitations

See [`docs/failure_modes.md`](docs/failure_modes.md) for a full breakdown. Key limits:

- **BM25 full rebuild** — every `add_chunks()` rebuilds the entire inverted index. At >100K docs, use Elasticsearch instead.
- **FAISS HNSW in RAM** — ~8M vectors on 16 GB RAM. For billion-scale, use Qdrant or DiskANN.
- **Cross-encoder latency** — ~800–1000 ms on CPU per query. Use MPS (Apple Silicon) or GPU to reduce to ~200 ms.
- **Self-check faithfulness** — heuristic proxy, ~80% precision. Not suitable as a production safety gate.

## Docker

```bash
cd deployment
docker compose up --build
```

The API will be available at `http://localhost:80` via Nginx.
