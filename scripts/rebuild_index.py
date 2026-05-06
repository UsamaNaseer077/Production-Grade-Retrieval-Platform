#!/usr/bin/env python3
"""
Rebuild the FAISS + BM25 index from all files already tracked in the hash store.

This fixes the case where the hash store was updated but the vector/BM25 index
was never written (e.g. a prior run was interrupted before the save step).

Usage:
    python scripts/rebuild_index.py [--dir data/shakespeare] [--force]

    --force   wipe the hash store before rebuild (treats all files as new)
    --dir     directory to rebuild from (default: all dirs in hash store)
"""
import argparse
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
from src.ingestion.pipeline import HashStore, IngestPipeline, SUPPORTED, _norm

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rebuild")


def rebuild(dirs, force=False):
    # Load indexes
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
    hs  = HashStore(CFG.index.hash_path)

    logger.info("Before rebuild — vectors=%d  bm25=%d  hashes=%d",
                vs.size, bm.size, hs.count())

    # Determine files to process
    if force:
        # Clear the hash store so every file is treated as new
        hs_path = Path(CFG.index.hash_path)
        hs_path.write_text("{}")
        hs = HashStore(CFG.index.hash_path)

    chunker = Chunker(
        strategy=CFG.chunking.strategy,
        child_size=CFG.chunking.child_size,
        parent_size=CFG.chunking.parent_size,
        overlap=CFG.chunking.overlap,
    )

    total_docs = 0
    total_chunks = 0

    for d in dirs:
        d = Path(d)
        if not d.exists():
            logger.warning("Directory not found: %s", d)
            continue

        # Force-ingest by temporarily removing the file hashes
        for f in sorted(d.glob("**/*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in SUPPORTED:
                continue

            # Remove this file's hash so it re-ingests
            import hashlib
            did = hashlib.sha256(f.read_bytes()).hexdigest()
            hs.remove(did)

        pip = IngestPipeline(hs)
        docs = pip.ingest_dir(d, recursive=True)
        if docs:
            chunks = chunker.chunk_many(docs)
            da = vs.add_chunks(chunks)
            sa = bm.add_chunks(chunks)
            logger.info("%s → %d docs, %d chunks, dense+%d, bm25+%d",
                        d, len(docs), len(chunks), da, sa)
            total_docs += len(docs)
            total_chunks += len(chunks)
        else:
            logger.info("%s → no new docs", d)

    vs.save()
    bm.save()
    hs.save()

    logger.info("Rebuild complete — vectors=%d  bm25=%d  hashes=%d  (added %d docs, %d chunks)",
                vs.size, bm.size, hs.count(), total_docs, total_chunks)
    return vs.size, bm.size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+",
                    default=["data/sample_docs", "data/shakespeare"],
                    help="Directories to rebuild from")
    ap.add_argument("--force", action="store_true",
                    help="Clear hash store before rebuild (full re-index)")
    args = ap.parse_args()
    rebuild(args.dirs, args.force)
