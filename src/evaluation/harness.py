import json, logging, time
from pathlib import Path
from statistics import mean, median, quantiles

logger = logging.getLogger(__name__)

def recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    return sum(1 for c in retrieved[:k] if c in relevant) / len(relevant)

def mrr(retrieved, relevant):
    for i,c in enumerate(retrieved,1):
        if c in relevant: return 1.0/i
    return 0.0

def run_eval(retriever, queries_path, qrels_path, k_values=None):
    k_values = k_values or [1,3,5,10]
    queries  = json.loads(Path(queries_path).read_text())
    qrels    = json.loads(Path(qrels_path).read_text())

    recall = {k:[] for k in k_values}
    rrs, lats, rows = [], [], []

    for q in queries:
        qid, text = q["query_id"], q["text"]
        rel = set(qrels.get(qid, []))
        if not rel: continue
        t0 = time.perf_counter()
        results = retriever.search(text)
        lat = (time.perf_counter()-t0)*1000
        lats.append(lat)
        ids = [r.chunk_id for r in results]
        rr  = mrr(ids, rel); rrs.append(rr)
        row = {"query_id":qid,"text":text,"latency_ms":round(lat,1),"mrr":rr}
        for k in k_values:
            v = recall_at_k(ids, rel, k); recall[k].append(v); row[f"recall@{k}"] = v
        rows.append(row)

    def pct(data, p):
        return round(quantiles(data,n=100)[p-1],1) if len(data)>=2 else (round(data[0],1) if data else 0)

    report = {
        "num_queries": len(rows),
        "mrr": round(mean(rrs),4) if rrs else 0,
        "latency": {
            "p50_ms": pct(lats,50), "p95_ms": pct(lats,95),
            "p99_ms": pct(lats,99), "mean_ms": round(mean(lats),1) if lats else 0,
        },
        "per_query": rows,
    }
    for k in k_values:
        report[f"recall@{k}"] = round(mean(recall[k]),4) if recall[k] else 0
    return report

def print_report(report):
    print("\n── Evaluation Report ──")
    print(f"  Queries : {report['num_queries']}")
    print(f"  MRR     : {report['mrr']}")
    for k in sorted(k for k in report if k.startswith("recall@")):
        print(f"  {k}   : {report[k]}")
    lat = report["latency"]
    print(f"  Lat p50 : {lat['p50_ms']} ms")
    print(f"  Lat p95 : {lat['p95_ms']} ms")
    print(f"  Lat p99 : {lat['p99_ms']} ms")
