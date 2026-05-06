import logging, time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunk_id: str; doc_id: str; parent_id: str
    text: str; parent_text: str; source: str
    metadata: dict
    # rrf_score: pre-rerank RRF fusion score (always positive, 1/(k+rank))
    # final_score: cross-encoder logit if reranker available, else same as rrf_score
    # Cross-encoder logits are unbounded and can be negative — this is expected.
    rrf_score: float = 0.0
    final_score: float = 0.0
    dense_rank: int = -1; sparse_rank: int = -1
    reranked: bool = False
    latency_ms: float = 0.0

def _rrf(dense, sparse, k=60):
    scores, meta_map = {}, {}
    for rank,doc in enumerate(dense):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid,0.0) + 1.0/(k+rank+1)
        meta_map[cid] = {**doc, "dense_rank": rank}
    for rank,doc in enumerate(sparse):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid,0.0) + 1.0/(k+rank+1)
        if cid not in meta_map: meta_map[cid] = doc
        meta_map[cid]["sparse_rank"] = rank
    return sorted(scores.items(), key=lambda x:x[1], reverse=True), meta_map

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None; self._ok = False
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name); self._ok = True
            logger.info("Cross-encoder loaded: %s", model_name)
        except Exception as e:
            logger.warning("Cross-encoder unavailable (%s) — RRF only", e)

    def rerank(self, query, candidates, top_k=10):
        """
        Returns candidates sorted by cross-encoder score.
        Each candidate is (chunk_id, rrf_score, meta).
        Returns (chunk_id, rrf_score, ce_score, meta) tuples.
        Falls back to RRF-sorted order (ce_score = rrf_score) if unavailable.
        """
        if not self._ok or not candidates:
            return [(cid, sc, sc, meta) for cid, sc, meta in candidates[:top_k]]
        pairs = [(query, c[2].get("text","")) for c in candidates]
        try:
            ce_scores = self._model.predict(pairs)
            ranked = sorted(
                zip(ce_scores, candidates),
                key=lambda x: x[0], reverse=True
            )
            return [(cid, rrf_sc, float(ce_sc), meta)
                    for ce_sc, (cid, rrf_sc, meta) in ranked[:top_k]]
        except Exception as e:
            logger.warning("Rerank failed (%s)", e)
            return [(cid, sc, sc, meta) for cid, sc, meta in candidates[:top_k]]

class HybridRetriever:
    def __init__(self, vs, bm25, reranker,
                 top_k_dense=50, top_k_sparse=50, top_k_rerank=10, rrf_k=60):
        self.vs=vs; self.bm25=bm25; self.reranker=reranker
        self.tkd=top_k_dense; self.tks=top_k_sparse
        self.tkr=top_k_rerank; self.rk=rrf_k

    def search(self, query):
        t0 = time.perf_counter()
        dense  = self.vs.search(query, self.tkd)
        sparse = self.bm25.search(query, self.tks)
        ranked, meta_map = _rrf(dense, sparse, self.rk)
        candidates = [(cid, sc, meta_map[cid]) for cid, sc in ranked]
        reranked   = self.reranker.rerank(query, candidates, self.tkr)
        total_ms   = (time.perf_counter()-t0)*1000
        did_rerank = self.reranker._ok
        out = []
        for cid, rrf_sc, final_sc, meta in reranked:
            out.append(RetrievalResult(
                chunk_id=cid, doc_id=meta.get("doc_id",""),
                parent_id=meta.get("parent_id",""), text=meta.get("text",""),
                parent_text=meta.get("parent_text", meta.get("text","")),
                source=meta.get("source",""), metadata=meta.get("metadata",{}),
                rrf_score=rrf_sc, final_score=final_sc,
                dense_rank=meta.get("dense_rank",-1),
                sparse_rank=meta.get("sparse_rank",-1),
                reranked=did_rerank, latency_ms=total_ms,
            ))
        return out
