#!/usr/bin/env python3
"""
Generate a realistic 1000+ document corpus for the retrieval platform.

Output layout
─────────────
  data/corpus/tech/           60 .txt files — software engineering topics
  data/corpus/science/        30 .txt files — natural sciences
  data/corpus/business/       20 .txt files — business/economics
  data/corpus/records/        40 .csv files — structured data records
  data/corpus/knowledge/      15 .json files — nested knowledge bases
  data/corpus/notes/          20 .md files — markdown notes

Total: ~185 source documents → 1,500+ chunks with parent_child chunking
(child_size=256, parent_size=1024 → ~6 children per 1500-word doc)

Eval queries
────────────
  eval/qrels.json   — chunk-level relevance judgments built from known content

Usage
─────
  python scripts/generate_corpus.py
  python scripts/run.py ingest --dir data/corpus
  python scripts/run.py eval
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RNG = random.Random(2024)

# ── content blocks by domain ──────────────────────────────────────────────────

TECH_TOPICS = {
    "information_retrieval": [
        "BM25 (Best Match 25) is a probabilistic ranking function used in information retrieval. "
        "It extends TF-IDF by incorporating document length normalization and a term saturation "
        "parameter k1. The formula scores documents by computing a weighted sum over query terms, "
        "where each term's contribution depends on its frequency in the document and the collection. "
        "BM25 remains competitive with neural methods on keyword-heavy queries where exact term "
        "matching is critical. The okapi variant is the most widely used.",

        "Dense retrieval encodes queries and documents into continuous vector embeddings using "
        "transformer models. The bi-encoder architecture encodes query and document independently, "
        "enabling offline indexing of all document embeddings. At query time, a fast approximate "
        "nearest-neighbour search (e.g. FAISS HNSW) finds top-k candidates. Dense retrieval excels "
        "at semantic similarity and paraphrase matching but can miss exact keyword matches that "
        "BM25 catches trivially.",

        "Hybrid retrieval combines sparse BM25 and dense embedding search to capture both lexical "
        "and semantic relevance. Reciprocal Rank Fusion (RRF) merges the two ranked lists without "
        "requiring score normalisation: each candidate's score is the sum of 1/(k+rank) across "
        "both lists, where k=60 is a smoothing constant. RRF consistently outperforms individual "
        "retrievers and weighted score combinations in head-to-head benchmarks on BEIR.",

        "Cross-encoder re-ranking applies a full attention model over (query, document) pairs "
        "to produce a fine-grained relevance score. Unlike bi-encoders, cross-encoders see both "
        "texts simultaneously, enabling deeper interaction. The ms-marco-MiniLM-L-6-v2 model "
        "achieves near-SOTA retrieval quality at modest inference cost. In a two-stage pipeline, "
        "the retriever fetches 50-100 candidates and the cross-encoder selects the top-10.",

        "Parent-child chunking embeds small 256-token child chunks for precise retrieval while "
        "returning the larger 1024-token parent chunk as context. This balances retrieval precision "
        "(small chunks are easier to match) with context quality (large chunks give coherent answers). "
        "The chunk_id encodes the document hash and parent/child indices, enabling O(1) parent lookup.",

        "Embedding caches store pre-computed vectors keyed by SHA-256(model_tag + text). On restart, "
        "only new or changed texts incur inference cost. A sharded directory layout (2-char prefix) "
        "prevents inode exhaustion at 1M+ entries. Cache hit rates above 90% are typical after "
        "the first full ingest, reducing warm-start latency by 100x vs cold encode.",
    ],
    "machine_learning": [
        "Transformer architectures use self-attention to model long-range dependencies in sequences. "
        "The attention mechanism computes query-key dot products scaled by sqrt(d_k) to prevent "
        "gradient vanishing in deep networks. Multi-head attention runs h parallel attention functions "
        "whose outputs are concatenated and projected. BERT pretrains with masked language modelling "
        "and next sentence prediction on large unlabelled corpora.",

        "Gradient descent minimises a loss function by computing the gradient with respect to all "
        "parameters and taking a step in the negative gradient direction. Stochastic gradient descent "
        "uses mini-batches to approximate the true gradient, reducing memory cost. Momentum, Adam, "
        "and AdaFactor are adaptive variants that adjust learning rates per-parameter based on "
        "gradient history, improving convergence on ill-conditioned objectives.",

        "Overfitting occurs when a model achieves low training loss but high validation loss by "
        "memorising noise in the training set rather than learning generalisable patterns. Regularisation "
        "techniques include L1 and L2 weight penalties, dropout (randomly zeroing activations), "
        "data augmentation, early stopping, and ensembling. The bias-variance trade-off quantifies "
        "the tension between model underfitting (high bias) and overfitting (high variance).",

        "Transfer learning reuses model weights pretrained on a large source task for a smaller "
        "downstream task. Fine-tuning updates all parameters on the target dataset, while parameter-"
        "efficient methods (LoRA, prefix tuning, adapters) add a small number of trainable parameters "
        "while freezing the backbone. Few-shot and zero-shot learning apply pretrained models directly "
        "without gradient updates via prompt engineering.",

        "Retrieval-augmented generation (RAG) grounds LLM responses in retrieved documents, reducing "
        "hallucination and enabling knowledge updates without retraining. The generator conditions on "
        "the concatenation of the query and top-k retrieved passages. Self-check faithfulness scoring "
        "detects when the generated answer diverges from the provided context by measuring token "
        "overlap between claims and retrieved passages.",

        "Contrastive learning trains encoders to bring representations of similar pairs closer "
        "while pushing dissimilar pairs apart. SimCSE uses in-batch negatives and dropout as a "
        "natural augmentation. Hard negative mining from BM25 or a weaker retriever significantly "
        "improves retrieval quality. The InfoNCE loss is a practical implementation of maximum "
        "likelihood estimation over the contrastive objective.",
    ],
    "distributed_systems": [
        "Consistent hashing distributes keys across nodes in a ring topology so that only K/N keys "
        "move when a node joins or leaves (K=keys, N=nodes). Virtual nodes (vnodes) improve load "
        "balance by placing each physical node at multiple positions in the ring. Cassandra, Riak, "
        "and Amazon DynamoDB use consistent hashing for horizontal scalability without a centralised "
        "coordinator.",

        "The CAP theorem states that a distributed system can provide at most two of: Consistency "
        "(every read receives the most recent write), Availability (every request receives a response), "
        "and Partition tolerance (the system continues operating despite network partitions). In "
        "practice, partitions are unavoidable in WANs so systems trade off between CP (HBase, Zookeeper) "
        "and AP (Cassandra, CouchDB).",

        "Event sourcing stores state as an append-only log of events rather than mutable records. "
        "The current state is derived by replaying all events from the beginning or from a snapshot. "
        "CQRS (Command Query Responsibility Segregation) separates the write model (commands, events) "
        "from the read model (projections, queries). Kafka is commonly used as the durable event log.",

        "Service mesh architectures (Istio, Linkerd) offload cross-cutting concerns — mutual TLS, "
        "observability, traffic management — from application code into a sidecar proxy. The control "
        "plane configures proxies centrally; the data plane intercepts all traffic. This enables "
        "canary deployments, circuit breakers, and rate limiting without code changes.",

        "Raft consensus achieves fault-tolerant distributed agreement through leader election, "
        "log replication, and safety properties. A leader is elected by majority vote; entries are "
        "committed only after replication to a majority of nodes. Followers that fall behind re-sync "
        "by requesting missing log entries. etcd and TiKV use Raft for distributed coordination.",

        "Distributed tracing instruments microservice calls with span context propagated via HTTP "
        "headers (W3C TraceContext standard). Each service creates child spans linked to the parent. "
        "Aggregated traces reveal end-to-end latency breakdown, identify bottlenecks, and surface "
        "cascading failures invisible to per-service metrics. Jaeger and Zipkin are popular backends.",
    ],
    "databases": [
        "B-tree indexes organise keys in a balanced tree with branching factor B. Searches, inserts, "
        "and deletes all run in O(log_B N) I/O operations. The B+ tree variant stores all values "
        "in leaf nodes linked in a sorted doubly-linked list, enabling efficient range scans. "
        "PostgreSQL, MySQL, and SQLite all use B+ trees as the default index structure.",

        "LSM trees (Log-Structured Merge-Trees) write all mutations to an in-memory memtable and "
        "an append-only WAL, then periodically flush sorted runs to disk (SSTables). Background "
        "compaction merges overlapping runs to bound read amplification. LSM trees excel at "
        "write-heavy workloads; RocksDB, LevelDB, and Apache Cassandra use LSM.",

        "Vector databases store high-dimensional embeddings and support approximate nearest-neighbour "
        "queries via HNSW, IVF, or Product Quantisation indexes. FAISS implements multiple ANN "
        "algorithms with GPU support. Qdrant, Weaviate, and Pinecone add metadata filtering, "
        "CRUD operations, and production features. At billion-scale, DiskANN and SPANN trade "
        "some recall for dramatically reduced memory footprint.",

        "SQL window functions compute aggregate values over a sliding partition of rows without "
        "collapsing them. PARTITION BY groups rows; ORDER BY sorts within the partition; the frame "
        "clause specifies which rows to include in each aggregate (ROWS BETWEEN, RANGE BETWEEN). "
        "Rank, dense_rank, row_number, lead, lag, and cumulative sum are common window functions.",

        "ACID transactions guarantee Atomicity (all-or-nothing), Consistency (integrity constraints "
        "preserved), Isolation (concurrent transactions behave as if serial), and Durability (committed "
        "data survives failure). PostgreSQL implements MVCC for snapshot isolation, reducing lock "
        "contention. Serialisable isolation detects write skew anomalies via predicate locking or "
        "serialisability conflict detection graphs.",

        "Database normalisation eliminates redundancy by decomposing tables into relations with "
        "minimal functional dependencies. First normal form requires atomic column values. Second "
        "normal form removes partial dependencies on composite primary keys. Third normal form "
        "removes transitive dependencies. BCNF (Boyce-Codd) is a stricter variant. Over-normalisation "
        "can hurt query performance; denormalisation trades space for read speed.",
    ],
    "software_engineering": [
        "Test-driven development (TDD) writes tests before implementation: red (failing test), green "
        "(minimal implementation), refactor (clean code). Tests serve as living documentation and "
        "prevent regression. Property-based testing (Hypothesis, QuickCheck) generates random inputs "
        "to find edge cases that hand-written tests miss. Mutation testing checks test suite quality "
        "by introducing deliberate bugs.",

        "Continuous integration runs automated builds and tests on every commit to catch integration "
        "failures early. GitHub Actions, GitLab CI, and Jenkins define workflows as YAML pipelines. "
        "Trunk-based development merges to main frequently in small increments; feature flags "
        "decouple deployment from release. Branch protection rules enforce CI passes before merging.",

        "Docker packages applications with their dependencies into portable images layered on a "
        "union filesystem. Dockerfile instructions create image layers; each RUN command adds a layer. "
        "Multi-stage builds compile in a build container and copy only runtime artifacts to the "
        "final image, minimising image size. Docker Compose orchestrates multi-container apps locally.",

        "API versioning strategies include URL path versioning (/v1/resource), header versioning "
        "(Accept: application/vnd.api+json;version=1), and query parameter versioning (?v=1). "
        "Semantic versioning uses MAJOR.MINOR.PATCH to communicate breaking changes. OpenAPI / Swagger "
        "generates interactive documentation from annotated route handlers.",

        "Clean Architecture organises code in concentric rings: entities (business rules), use cases "
        "(application logic), interface adapters (controllers/presenters), and infrastructure "
        "(databases, frameworks). Dependencies point inward; outer layers depend on inner abstractions "
        "not implementations. This decouples business logic from framework details, enabling "
        "independent testing of each layer.",

        "Observability in production requires three pillars: metrics (time-series counters/gauges "
        "via Prometheus), logs (structured JSON via ELK or Loki), and traces (distributed spans "
        "via Jaeger). The USE method (Utilisation, Saturation, Errors) guides resource analysis; "
        "the RED method (Rate, Errors, Duration) guides service health. SLOs bound acceptable "
        "latency percentiles and error rates.",
    ],
}

SCIENCE_TOPICS = {
    "physics": [
        "Quantum mechanics describes the behaviour of particles at atomic and subatomic scales. "
        "The Schrödinger equation governs how quantum states evolve in time. Wave-particle duality "
        "means electrons exhibit interference patterns in double-slit experiments yet arrive as "
        "discrete particles. The Heisenberg uncertainty principle bounds the product of position "
        "and momentum uncertainties: ΔxΔp ≥ ℏ/2.",

        "General relativity describes gravity as spacetime curvature caused by mass-energy. "
        "The Einstein field equations relate the spacetime metric to the stress-energy tensor. "
        "Gravitational waves — ripples in spacetime — were first detected by LIGO in 2015 from "
        "a binary black hole merger. Black holes form when mass collapses inside its Schwarzschild "
        "radius; the event horizon is the one-way boundary beyond which nothing escapes.",

        "Thermodynamics governs energy transformations. The first law states energy is conserved; "
        "the second law states entropy of an isolated system never decreases. The Carnot engine "
        "defines the maximum efficiency of a heat engine between temperatures T_hot and T_cold: "
        "η = 1 − T_cold/T_hot. Statistical mechanics derives thermodynamic laws from the "
        "probabilistic behaviour of large numbers of particles.",
    ],
    "biology": [
        "DNA replication is semi-conservative: each strand of the double helix serves as template "
        "for a new complementary strand. DNA polymerase synthesises new strands 5' to 3', requiring "
        "a primase-laid RNA primer. The leading strand is synthesised continuously; the lagging strand "
        "is synthesised in Okazaki fragments joined by DNA ligase. Proofreading reduces error rates "
        "to roughly one mistake per billion base pairs.",

        "CRISPR-Cas9 is a programmable genome editing system derived from bacterial immune systems. "
        "A guide RNA directs the Cas9 nuclease to a target sequence adjacent to a protospacer "
        "adjacent motif (PAM). Cas9 creates a double-strand break; cellular repair pathways introduce "
        "mutations (NHEJ) or precise edits (HDR). CRISPR has applications in treating genetic "
        "diseases, engineering crop plants, and basic research.",

        "Protein folding converts a linear amino acid sequence into a three-dimensional functional "
        "structure. Primary structure is the sequence; secondary structure forms alpha helices and "
        "beta sheets via hydrogen bonds; tertiary structure is the overall 3D fold; quaternary "
        "structure is the assembly of multiple chains. AlphaFold2 uses attention-based neural "
        "networks to predict protein structures from sequence with near-experimental accuracy.",
    ],
    "chemistry": [
        "Electrochemistry studies electron transfer reactions at electrode-electrolyte interfaces. "
        "Galvanic cells convert chemical energy to electrical energy via spontaneous redox reactions. "
        "The Nernst equation relates cell potential to ion concentrations. Lithium-ion batteries "
        "store energy in electrochemical potential between graphite anodes and lithium metal oxide "
        "cathodes; capacity is measured in milliampere-hours per gram.",

        "Catalysis lowers the activation energy of chemical reactions without being consumed. "
        "Heterogeneous catalysts (platinum, palladium) adsorb reactants onto active surface sites. "
        "Enzymes are biological catalysts that achieve extraordinary specificity and rate acceleration "
        "through substrate binding in active sites. Zeolites are microporous aluminosilicates used "
        "as acid catalysts in petroleum refining.",

        "Spectroscopy characterises molecules by their interaction with electromagnetic radiation. "
        "IR spectroscopy identifies functional groups by characteristic absorption frequencies. "
        "NMR spectroscopy determines molecular structure from nuclear spin resonances. Mass spectrometry "
        "measures mass-to-charge ratios of ionised molecules, identifying compound identities and "
        "isotope distributions. X-ray crystallography resolves 3D atomic structures by analysing "
        "diffraction patterns.",
    ],
}

BUSINESS_TOPICS = {
    "finance": [
        "The capital asset pricing model (CAPM) prices risky assets by relating expected return "
        "to systematic market risk (beta). Beta measures an asset's sensitivity to market movements. "
        "The Sharpe ratio measures risk-adjusted return: (portfolio return − risk-free rate) / "
        "standard deviation. Modern portfolio theory shows that diversification reduces idiosyncratic "
        "risk without necessarily reducing expected return.",

        "Discounted cash flow (DCF) values an asset by summing projected future cash flows discounted "
        "at a risk-adjusted rate. Net present value (NPV) is DCF minus initial investment; positive "
        "NPV projects create value. The internal rate of return (IRR) is the discount rate at which "
        "NPV equals zero. Weighted average cost of capital (WACC) blends the cost of equity and "
        "after-tax cost of debt proportionally.",

        "Options pricing models value the right (not obligation) to buy or sell an asset at a "
        "strike price by an expiration date. The Black-Scholes model assumes log-normal asset prices, "
        "constant volatility, and no arbitrage. Greeks (delta, gamma, theta, vega) measure "
        "sensitivity to price, curvature, time, and volatility. Implied volatility is the market's "
        "expectation of future price movement extracted from traded option prices.",
    ],
    "operations": [
        "Supply chain management coordinates the flow of goods, information, and money from raw "
        "material suppliers through manufacturers to end customers. Just-in-time (JIT) inventory "
        "reduces holding costs but requires reliable supplier lead times. The bullwhip effect "
        "amplifies demand variability upstream; information sharing and vendor-managed inventory "
        "dampen it. Total landed cost includes purchase price, freight, duties, and handling.",

        "Lean manufacturing eliminates the seven wastes: overproduction, waiting, transport, "
        "over-processing, inventory, motion, and defects. The five S's (Sort, Set in order, Shine, "
        "Standardise, Sustain) organise the workplace. Kaizen means continuous improvement through "
        "small incremental changes. Value stream mapping identifies bottlenecks and non-value-adding "
        "steps. Six Sigma uses DMAIC (Define, Measure, Analyse, Improve, Control) to reduce defects.",

        "Game theory models strategic interaction between rational agents. Nash equilibrium is a "
        "state where no player benefits from unilaterally changing strategy. The prisoner's dilemma "
        "shows that individual rationality can produce collectively suboptimal outcomes. Mechanism "
        "design (reverse game theory) constructs rules that align individual incentives with "
        "social objectives, used in auction design and market regulation.",
    ],
}


# ── generators ────────────────────────────────────────────────────────────────

def _paragraphs(blocks, n=4):
    """Pick n random blocks and combine into a multi-paragraph document."""
    chosen = RNG.sample(blocks, min(n, len(blocks)))
    RNG.shuffle(chosen)
    return "\n\n".join(chosen)


def _expand(text, reps=2):
    """Expand text by concatenating slight variations to reach ~1500+ words."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    parts = [text]
    for _ in range(reps):
        RNG.shuffle(sentences)
        parts.append(". ".join(sentences[:len(sentences)//2]) + ".")
    return "\n\n".join(parts)


def write_tech_docs(out_dir: Path) -> list:
    """Write 60 technical text documents (~1800 words each)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for topic, blocks in TECH_TOPICS.items():
        for i in range(10):
            # Each file = 4 random blocks × 2 repetitions ≈ 1800 words
            para = _paragraphs(blocks, n=4)
            text = _expand(para, reps=2)
            fname = f"{topic}_{i+1:02d}.txt"
            path = out_dir / fname
            path.write_text(f"# {topic.replace('_', ' ').title()} — Part {i+1}\n\n{text}\n",
                            encoding="utf-8")
            paths.append(path)
    return paths


def write_science_docs(out_dir: Path) -> list:
    """Write 30 science text documents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for topic, blocks in SCIENCE_TOPICS.items():
        for i in range(10):
            para = _paragraphs(blocks, n=min(3, len(blocks)))
            text = _expand(para, reps=3)
            fname = f"{topic}_{i+1:02d}.txt"
            path = out_dir / fname
            path.write_text(f"# {topic.replace('_', ' ').title()} — Part {i+1}\n\n{text}\n",
                            encoding="utf-8")
            paths.append(path)
    return paths


def write_business_docs(out_dir: Path) -> list:
    """Write 20 business text documents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for topic, blocks in BUSINESS_TOPICS.items():
        for i in range(10):
            para = _paragraphs(blocks, n=min(3, len(blocks)))
            text = _expand(para, reps=3)
            fname = f"{topic}_{i+1:02d}.txt"
            path = out_dir / fname
            path.write_text(f"# {topic.replace('_', ' ').title()} — Part {i+1}\n\n{text}\n",
                            encoding="utf-8")
            paths.append(path)
    return paths


def write_csv_records(out_dir: Path) -> list:
    """Write 40 CSV files with structured records (100 rows each)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    templates = [
        # (filename, headers, row_generator)
        ("algorithms.csv",
         ["algorithm", "type", "time_complexity", "space_complexity", "use_case"],
         lambda i: [
             RNG.choice(["BM25","HNSW","LSM-Tree","B-Tree","QuickSort","MergeSort","DFS","BFS",
                         "Dijkstra","Floyd-Warshall","KMP","Rabin-Karp","RAFT","Paxos"]),
             RNG.choice(["search","sort","graph","string","consensus","tree"]),
             RNG.choice(["O(1)","O(log n)","O(n)","O(n log n)","O(n²)","O(2^n)"]),
             RNG.choice(["O(1)","O(n)","O(n log n)","O(n²)"]),
             RNG.choice(["retrieval","indexing","compression","routing","replication"]),
         ]),
        ("ml_papers.csv",
         ["title", "year", "venue", "topic", "citation_count"],
         lambda i: [
             RNG.choice(["Attention Is All You Need","BERT","GPT-3","RoBERTa","T5",
                         "DPR","ColBERT","ANCE","REALM","RAG","InstructGPT",
                         "LLaMA","Alpaca","LoRA","QLoRA","FLAN","PaLM","Chinchilla"]),
             RNG.randint(2017, 2024),
             RNG.choice(["NeurIPS","ICML","ICLR","ACL","EMNLP","NAACL","SIGIR","CIKM"]),
             RNG.choice(["language_model","retrieval","generation","classification","embeddings"]),
             RNG.randint(50, 50000),
         ]),
        ("software_metrics.csv",
         ["service", "latency_p50_ms", "latency_p99_ms", "error_rate_pct",
          "throughput_rps", "availability_pct"],
         lambda i: [
             RNG.choice(["retrieval","embedding","reranker","bm25","llm","cache",
                         "ingest","api","db","memory"]),
             round(RNG.uniform(1, 100), 1),
             round(RNG.uniform(10, 2000), 1),
             round(RNG.uniform(0, 2), 3),
             RNG.randint(10, 10000),
             round(RNG.uniform(99.0, 99.999), 3),
         ]),
        ("research_topics.csv",
         ["topic", "subtopic", "open_problem", "key_paper", "maturity"],
         lambda i: [
             RNG.choice(["NLP","CV","RL","IR","RecSys","MLSys","Robotics","Bioinformatics"]),
             RNG.choice(["efficiency","interpretability","robustness","fairness","scalability"]),
             RNG.choice(["sample efficiency","distribution shift","long-range dependencies",
                         "factual grounding","reasoning","calibration"]),
             RNG.choice(["transformer","BERT","GPT","ResNet","CLIP","DALL-E","Codex"]),
             RNG.choice(["research","early","mature","production","legacy"]),
         ]),
    ]

    for fname, headers, row_fn in templates:
        for batch in range(10):
            path = out_dir / f"{fname.replace('.csv', '')}_{batch+1:02d}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(headers)
                for i in range(100):
                    w.writerow(row_fn(i))
            paths.append(path)

    return paths


def write_json_knowledge(out_dir: Path) -> list:
    """Write 15 JSON knowledge base files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    bases = [
        {
            "name": "retrieval_concepts",
            "entries": [
                {"concept": "BM25", "type": "sparse_retrieval",
                 "description": "Probabilistic ranking using term frequency, document frequency, and "
                                "document length normalisation. Controlled by k1 (term saturation) "
                                "and b (length normalisation) hyperparameters.",
                 "strengths": ["exact keyword match", "no GPU required", "interpretable"],
                 "weaknesses": ["no semantic understanding", "vocabulary mismatch"]},
                {"concept": "Dense Retrieval", "type": "neural_retrieval",
                 "description": "Bi-encoder architecture encodes query and document independently. "
                                "FAISS or ScaNN for approximate nearest-neighbour search at scale.",
                 "strengths": ["semantic matching", "paraphrase recall"],
                 "weaknesses": ["no exact match", "computationally expensive", "stale if not updated"]},
                {"concept": "RRF", "type": "fusion",
                 "description": "Reciprocal Rank Fusion score = sum(1/(k+rank)) across lists. "
                                "k=60 is empirically robust. Requires no score normalisation.",
                 "strengths": ["robust", "no calibration needed"],
                 "weaknesses": ["ignores score magnitude"]},
                {"concept": "Cross-Encoder Reranker", "type": "reranking",
                 "description": "Full attention over (query, document) joint encoding. Slower "
                                "than bi-encoder but more accurate. Used in second-stage re-ranking "
                                "of top-50 candidates.",
                 "strengths": ["highest precision", "deep interaction"],
                 "weaknesses": ["cannot index offline", "high inference latency at scale"]},
            ],
        },
        {
            "name": "llm_backends",
            "entries": [
                {"backend": "Ollama", "type": "local",
                 "models": ["llama3.2", "mistral", "phi3", "gemma"],
                 "notes": "Requires `ollama serve`. Free, private, no API key.",
                 "use_case": "development, air-gapped environments"},
                {"backend": "OpenAI", "type": "cloud",
                 "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                 "notes": "Requires OPENAI_API_KEY. Best quality at cost.",
                 "use_case": "production, high-quality answers"},
                {"backend": "Anthropic", "type": "cloud",
                 "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
                 "notes": "Requires ANTHROPIC_API_KEY. Strong instruction following.",
                 "use_case": "production, safety-critical applications"},
            ],
        },
        {
            "name": "chunking_strategies",
            "entries": [
                {"strategy": "parent_child", "child_size": 256, "parent_size": 1024,
                 "description": "Embed small child chunks; return larger parent for context. "
                                "Best for long documents where precision matters more than context.",
                 "recommended_for": "RAG, question answering"},
                {"strategy": "sliding", "overlap": 64,
                 "description": "Overlapping fixed windows. Higher recall on long documents "
                                "at the cost of index size.",
                 "recommended_for": "dense technical documents"},
                {"strategy": "structural", "split_on": "headings",
                 "description": "Split on Markdown H1-H3 boundaries from docling output. "
                                "Preserves semantic sections.",
                 "recommended_for": "structured documentation, PDFs"},
            ],
        },
    ]

    for i, base in enumerate(bases):
        path = out_dir / f"{base['name']}.json"
        path.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(path)

    # Generate random variations
    for j in range(12):
        data = {
            "id": f"kb_{j+1:03d}",
            "domain": RNG.choice(["computer_science", "physics", "biology", "finance", "operations"]),
            "entries": [
                {
                    "term": RNG.choice([
                        "entropy", "gradient", "eigenvector", "Markov chain", "Bayes theorem",
                        "convolution", "attention", "embedding", "latency", "throughput",
                        "consistency", "availability", "sharding", "indexing", "caching",
                        "pipeline", "parallelism", "fault tolerance", "backpressure", "idempotency",
                    ]),
                    "definition": RNG.choice([
                        "A fundamental concept characterising information, uncertainty, and disorder.",
                        "A mathematical operator that iteratively updates model parameters.",
                        "A vector that remains collinear after a linear transformation.",
                        "A stochastic process where future state depends only on current state.",
                        "A framework for updating beliefs given new evidence.",
                        "An operation that computes a weighted sum over a sliding window.",
                        "A mechanism for computing context-dependent representations.",
                        "A dense vector representation in a learned feature space.",
                    ]),
                    "examples": [f"Example {k+1}: application in domain context." for k in range(3)],
                }
                for _ in range(RNG.randint(5, 15))
            ],
        }
        path = out_dir / f"knowledge_base_{j+1:03d}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        paths.append(path)

    return paths


def write_markdown_notes(out_dir: Path) -> list:
    """Write 20 Markdown notes with heading structure (for structural chunking demo)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    topics_blocks = list(TECH_TOPICS.items()) + list(SCIENCE_TOPICS.items())

    for i in range(20):
        topic_name, blocks = RNG.choice(topics_blocks)
        chosen = RNG.sample(blocks, min(4, len(blocks)))
        content = f"# {topic_name.replace('_',' ').title()} — Study Notes\n\n"
        subtitles = ["Overview", "Key Concepts", "Implementation Details",
                     "Trade-offs and Limitations", "Real-World Applications", "Summary"]
        for j, (subtitle, block) in enumerate(zip(subtitles, chosen)):
            content += f"## {subtitle}\n\n{block}\n\n"
            # Add a sub-section
            sentences = block.split(". ")
            if len(sentences) > 2:
                content += f"### Details\n\n{'. '.join(sentences[:2])}.\n\n"
        path = out_dir / f"notes_{i+1:02d}.md"
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    return paths


# ── qrels builder ─────────────────────────────────────────────────────────────

def build_qrels(queries_path: Path, retriever) -> dict:
    """
    Build qrels by running each query and taking top results that match
    the expected source pattern. Returns {query_id: [chunk_id, ...]} mapping.
    """
    from src.ingestion.pipeline import _norm
    queries = json.loads(queries_path.read_text())
    qrels = {}
    for q in queries:
        qid = q["query_id"]
        pattern = q.get("expected_source", "")
        results = retriever.search(q["text"])
        if pattern:
            # Only accept chunks from the expected source
            relevant = [r.chunk_id for r in results
                        if pattern.lower() in r.source.lower()][:3]
        else:
            relevant = [r.chunk_id for r in results[:3]]
        qrels[qid] = relevant
    return qrels


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    corpus_root = Path("data/corpus")
    print("Generating corpus …")

    txt_paths = write_tech_docs(corpus_root / "tech")
    print(f"  Tech docs:     {len(txt_paths)} files → {corpus_root / 'tech'}")

    sci_paths = write_science_docs(corpus_root / "science")
    print(f"  Science docs:  {len(sci_paths)} files → {corpus_root / 'science'}")

    biz_paths = write_business_docs(corpus_root / "business")
    print(f"  Business docs: {len(biz_paths)} files → {corpus_root / 'business'}")

    csv_paths = write_csv_records(corpus_root / "records")
    print(f"  CSV records:   {len(csv_paths)} files → {corpus_root / 'records'}")

    json_paths = write_json_knowledge(corpus_root / "knowledge")
    print(f"  JSON KB:       {len(json_paths)} files → {corpus_root / 'knowledge'}")

    md_paths = write_markdown_notes(corpus_root / "notes")
    print(f"  Markdown:      {len(md_paths)} files → {corpus_root / 'notes'}")

    total = len(txt_paths) + len(sci_paths) + len(biz_paths) + len(csv_paths) + len(json_paths) + len(md_paths)
    total_bytes = sum(p.stat().st_size for p in
                      txt_paths + sci_paths + biz_paths + csv_paths + json_paths + md_paths)
    print(f"\n  Total source files: {total}")
    print(f"  Total corpus size:  {total_bytes/1024:.0f} KB")
    print(f"\nNext steps:")
    print(f"  python scripts/run.py ingest --dir data/corpus")
    print(f"  python scripts/build_qrels.py")


if __name__ == "__main__":
    main()
