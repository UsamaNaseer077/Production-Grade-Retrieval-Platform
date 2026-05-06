# Information Retrieval — Study Notes

## Overview

Cross-encoder re-ranking applies a full attention model over (query, document) pairs to produce a fine-grained relevance score. Unlike bi-encoders, cross-encoders see both texts simultaneously, enabling deeper interaction. The ms-marco-MiniLM-L-6-v2 model achieves near-SOTA retrieval quality at modest inference cost. In a two-stage pipeline, the retriever fetches 50-100 candidates and the cross-encoder selects the top-10.

### Details

Cross-encoder re-ranking applies a full attention model over (query, document) pairs to produce a fine-grained relevance score. Unlike bi-encoders, cross-encoders see both texts simultaneously, enabling deeper interaction.

## Key Concepts

Parent-child chunking embeds small 256-token child chunks for precise retrieval while returning the larger 1024-token parent chunk as context. This balances retrieval precision (small chunks are easier to match) with context quality (large chunks give coherent answers). The chunk_id encodes the document hash and parent/child indices, enabling O(1) parent lookup.

### Details

Parent-child chunking embeds small 256-token child chunks for precise retrieval while returning the larger 1024-token parent chunk as context. This balances retrieval precision (small chunks are easier to match) with context quality (large chunks give coherent answers).

## Implementation Details

Embedding caches store pre-computed vectors keyed by SHA-256(model_tag + text). On restart, only new or changed texts incur inference cost. A sharded directory layout (2-char prefix) prevents inode exhaustion at 1M+ entries. Cache hit rates above 90% are typical after the first full ingest, reducing warm-start latency by 100x vs cold encode.

### Details

Embedding caches store pre-computed vectors keyed by SHA-256(model_tag + text). On restart, only new or changed texts incur inference cost.

## Trade-offs and Limitations

BM25 (Best Match 25) is a probabilistic ranking function used in information retrieval. It extends TF-IDF by incorporating document length normalization and a term saturation parameter k1. The formula scores documents by computing a weighted sum over query terms, where each term's contribution depends on its frequency in the document and the collection. BM25 remains competitive with neural methods on keyword-heavy queries where exact term matching is critical. The okapi variant is the most widely used.

### Details

BM25 (Best Match 25) is a probabilistic ranking function used in information retrieval. It extends TF-IDF by incorporating document length normalization and a term saturation parameter k1.

