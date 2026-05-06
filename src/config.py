import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lsa_dim: int = 128
    cache_dir: Path = ROOT / "data" / "emb_cache"   # KV cache — skip re-embed on restart

@dataclass
class ChunkingConfig:
    child_size: int = 256
    parent_size: int = 1024
    overlap: int = 32
    strategy: str = "parent_child"  # fixed | sentence | parent_child | sliding | structural

@dataclass
class IndexConfig:
    hnsw_m: int = 32
    index_path: Path = ROOT / "data" / "faiss.index"
    meta_path: Path  = ROOT / "data" / "meta.jsonl"
    hash_path: Path  = ROOT / "data" / "hashes.json"
    bm25_path: Path  = ROOT / "data" / "bm25.pkl"
    lsa_path: Path   = ROOT / "data" / "lsa.pkl"

@dataclass
class RetrievalConfig:
    top_k_dense: int  = 50
    top_k_sparse: int = 50
    top_k_rerank: int = 10
    rrf_k: int        = 60

@dataclass
class LLMConfig:
    # ── backend selection ───────────────────────────────────────────────────
    # "auto"        → try: ollama → openai → anthropic → huggingface → transformers
    # "ollama"      → local Ollama server (ollama serve + ollama pull <model>)
    # "openai"      → OpenAI API (set OPENAI_API_KEY env var)
    # "anthropic"   → Anthropic API (set ANTHROPIC_API_KEY env var)
    # "huggingface" → HF Inference API (set HF_API_TOKEN env var)
    # "transformers"→ local HuggingFace model, CPU-safe (downloads on first run)
    backend:     str   = "auto"
    # "auto" → picks the sensible default for the detected backend
    model_name:  str   = "auto"
    temperature: float = 0.0
    max_tokens:  int   = 512
    timeout:     int   = 60
    prompt_path: Path  = ROOT / "prompts" / "system.md"

@dataclass
class MemoryConfig:
    stm_max_turns: int = 10             # short-term: conversation ring buffer
    stm_ttl_seconds: int = 3600         # short-term: session TTL
    ltm_db_path: Path = ROOT / "data" / "memory.db"   # long-term: SQLite

@dataclass
class EvalConfig:
    k_values: list = field(default_factory=lambda: [1, 3, 5, 10])
    queries_path: Path = ROOT / "eval" / "queries.json"
    qrels_path: Path   = ROOT / "eval" / "qrels.json"

@dataclass
class AppConfig:
    embedding:  EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking:   ChunkingConfig  = field(default_factory=ChunkingConfig)
    index:      IndexConfig     = field(default_factory=IndexConfig)
    retrieval:  RetrievalConfig = field(default_factory=RetrievalConfig)
    llm:        LLMConfig       = field(default_factory=LLMConfig)
    memory:     MemoryConfig    = field(default_factory=MemoryConfig)
    ev:         EvalConfig      = field(default_factory=EvalConfig)

CFG = AppConfig()
