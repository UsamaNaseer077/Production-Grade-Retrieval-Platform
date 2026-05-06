# Information Retrieval — Study Notes

## Overview

Dense retrieval encodes queries and documents into continuous vector embeddings using transformer models. The bi-encoder architecture encodes query and document independently, enabling offline indexing of all document embeddings. At query time, a fast approximate nearest-neighbour search (e.g. FAISS HNSW) finds top-k candidates. Dense retrieval excels at semantic similarity and paraphrase matching but can miss exact keyword matches that BM25 catches trivially.

### Details

Dense retrieval encodes queries and documents into continuous vector embeddings using transformer models. The bi-encoder architecture encodes query and document independently, enabling offline indexing of all document embeddings.

## Key Concepts

Parent-child chunking embeds small 256-token child chunks for precise retrieval while returning the larger 1024-token parent chunk as context. This balances retrieval precision (small chunks are easier to match) with context quality (large chunks give coherent answers). The chunk_id encodes the document hash and parent/child indices, enabling O(1) parent lookup.

### Details

Parent-child chunking embeds small 256-token child chunks for precise retrieval while returning the larger 1024-token parent chunk as context. This balances retrieval precision (small chunks are easier to match) with context quality (large chunks give coherent answers).

## Implementation Details

Hybrid retrieval combines sparse BM25 and dense embedding search to capture both lexical and semantic relevance. Reciprocal Rank Fusion (RRF) merges the two ranked lists without requiring score normalisation: each candidate's score is the sum of 1/(k+rank) across both lists, where k=60 is a smoothing constant. RRF consistently outperforms individual retrievers and weighted score combinations in head-to-head benchmarks on BEIR.

### Details

Hybrid retrieval combines sparse BM25 and dense embedding search to capture both lexical and semantic relevance. Reciprocal Rank Fusion (RRF) merges the two ranked lists without requiring score normalisation: each candidate's score is the sum of 1/(k+rank) across both lists, where k=60 is a smoothing constant.

## Trade-offs and Limitations

BM25 (Best Match 25) is a probabilistic ranking function used in information retrieval. It extends TF-IDF by incorporating document length normalization and a term saturation parameter k1. The formula scores documents by computing a weighted sum over query terms, where each term's contribution depends on its frequency in the document and the collection. BM25 remains competitive with neural methods on keyword-heavy queries where exact term matching is critical. The okapi variant is the most widely used.

### Details

BM25 (Best Match 25) is a probabilistic ranking function used in information retrieval. It extends TF-IDF by incorporating document length normalization and a term saturation parameter k1.

