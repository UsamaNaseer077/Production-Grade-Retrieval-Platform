# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System libs needed by faiss, sentence-transformers, docling
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default embedding model so the container works offline
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    || echo "SentenceTransformer download failed — will use LSA fallback"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM deps AS runtime

WORKDIR /app

COPY src/       src/
COPY prompts/   prompts/
COPY eval/      eval/

# Persistent data (indexes, cache, memory) are mounted as volumes at runtime
RUN mkdir -p data/emb_cache data/sample_docs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Uvicorn with 2 workers per CPU core.
# For read-heavy retrieval workloads, 2-4 workers per vCPU is the sweet spot.
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
