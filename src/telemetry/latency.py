"""
Per-component latency tracker with rolling window percentiles.

Usage
-----
    from src.telemetry import TRACKER

    with TRACKER.track("dense_search"):
        results = vs.search(query)

    TRACKER.record("llm", llm_ms)

    print(TRACKER.all_stats())
    # {'embed': {'p50_ms': 12, 'p95_ms': 28, ...}, 'dense_search': {...}, ...}

Components tracked
------------------
  embed        — query embedding
  dense_search — FAISS HNSW search
  bm25_search  — BM25 keyword search
  rerank       — cross-encoder re-ranking
  llm          — LLM generation (token count not measured)
  self_check   — faithfulness check
  total        — full pipeline wall-clock
"""
import time
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from statistics import mean, quantiles
from typing import Dict, List

logger = logging.getLogger(__name__)

WINDOW = 1000   # rolling window size per component


class LatencyTracker:
    def __init__(self, window: int = WINDOW):
        self._window = window
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._request_total: deque = deque(maxlen=window)

    # ── context manager ──────────────────────────────────────────────────────

    @contextmanager
    def track(self, component: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self._data[component].append(ms)

    # ── manual recording ─────────────────────────────────────────────────────

    def record(self, component: str, ms: float):
        self._data[component].append(ms)

    def record_request(self, ms: float):
        self._request_total.append(ms)

    # ── per-request breakdown ─────────────────────────────────────────────────

    @staticmethod
    def start() -> float:
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 1)

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, component: str) -> dict:
        return _compute(list(self._data.get(component, [])))

    def all_stats(self) -> dict:
        out: dict = {"total_request": _compute(list(self._request_total))}
        for comp, dq in self._data.items():
            out[comp] = _compute(list(dq))
        return out

    def summary_line(self) -> str:
        """One-liner for logging."""
        s = self.all_stats()
        parts = [f"{k}=p50:{v['p50_ms']}ms" for k, v in s.items() if v["count"]]
        return "  ".join(parts)


def _compute(data: List[float]) -> dict:
    if not data:
        return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "mean_ms": 0, "count": 0}

    def pct(p: int) -> float:
        if len(data) < 2:
            return round(data[0], 1)
        return round(quantiles(data, n=100)[p - 1], 1)

    return {
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": round(mean(data), 1),
        "count": len(data),
    }


# ── global singleton ─────────────────────────────────────────────────────────
TRACKER = LatencyTracker()
