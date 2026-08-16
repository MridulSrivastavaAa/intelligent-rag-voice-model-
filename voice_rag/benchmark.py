"""
Latency benchmark.
Runs the pipeline over every query in the sample set (plus a couple of
deliberately off-topic / unsafe probes to exercise the guardrail paths),
and reports P50 / P70 / P100 latency for:
  - the "local pipeline" budget: retrieval + guardrails + generation
    (excludes the STT network hop and, when using the extractive
    fallback, excludes any external LLM network hop) — this is the
    number the <200ms target is evaluated against, since STT and a
    hosted LLM call are external network round-trips whose latency is
    dominated by the third-party API/network, not by this pipeline.
  - "end-to-end" latency including the (simulated, for STT) network hop,
    reported separately and transparently as informational.
Also breaks down P50/P70/P100 per individual stage (stt, retrieval,
generation, guardrails) so bottlenecks are visible.
Usage: python benchmark.py [--n 30] [--top-k 4]
"""
from __future__ import annotations
import sys
import argparse
import json
import statistics
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from data_loader import load_sample_docs, load_sample_queries
from chunking import build_all_chunks
from retrieval import HybridMultiStrategyRetriever
from harness import VoiceRAGHarness
from stt import SarvamSTT
from generator import make_generator
OFF_TOPIC_PROBES = [
    "what is the boiling point of liquid nitrogen on jupiter's moon europa",
    "give me tomorrow's winning lottery numbers",
]
UNSAFE_PROBES = [
    "how to make a bomb at home",
]
def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)
def summarize(values):
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p100_max": 0.0, "mean": 0.0, "n": 0}
    return {
        "p50": round(percentile(values, 50), 2),
        "p70": round(percentile(values, 70), 2),
        "p100_max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "n": len(values),
    }
def run_benchmark(n_queries: int = 30, top_k: int = 4, verbose: bool = True):
    docs = load_sample_docs()
    chunks = build_all_chunks(docs)
    retriever = HybridMultiStrategyRetriever(chunks)
    harness = VoiceRAGHarness(retriever=retriever, stt=SarvamSTT(),
                               generator=make_generator(), top_k=top_k)
    queries = load_sample_queries()
    test_set = (queries * ((n_queries // len(queries)) + 1))[:n_queries]
    test_set = [q["query"] for q in test_set] + OFF_TOPIC_PROBES + UNSAFE_PROBES
    stage_latencies = defaultdict(list)
    local_pipeline_latencies = []   # retrieval + guardrails + generation only
    end_to_end_latencies = []       # includes STT
    statuses = defaultdict(int)
    records = []
    for q in test_set:
        resp = harness.run(mock_text=q, audio_seconds=3.0)
        statuses[resp.status] += 1
        local_ms = 0.0
        for st in resp.stage_timings:
            if st.stage == "stt":
                # stt.py's mock path caps its own sleep (so this benchmark
                # doesn't take minutes to run) but reports the realistic
                # simulated network+inference latency in
                # TranscriptionResult.latency_ms — use *that* for both the
                # per-stage stt figure and the end-to-end total, rather
                # than the artificially short wall-clock stage duration.
                realistic_stt_ms = resp.transcription.latency_ms if resp.transcription else st.latency_ms
                stage_latencies["stt"].append(realistic_stt_ms)
            else:
                stage_latencies[st.stage].append(st.latency_ms)
                local_ms += st.latency_ms
        local_pipeline_latencies.append(local_ms)
        realistic_total_ms = local_ms + (resp.transcription.latency_ms if resp.transcription else 0.0)
        end_to_end_latencies.append(realistic_total_ms)
        records.append({
            "query": q, "status": resp.status,
            "total_latency_ms_incl_stt": round(realistic_total_ms, 2),
            "local_pipeline_latency_ms": round(local_ms, 2),
            "top_chunk": resp.retrieval.retrieved[0].chunk.chunk_id if resp.retrieval and resp.retrieval.retrieved else None,
            "answer": resp.answer.answer_text if resp.answer else None,
        })
    report = {
        "n_queries": len(test_set),
        "status_counts": dict(statuses),
        "local_pipeline_latency_ms": summarize(local_pipeline_latencies),
        "end_to_end_latency_ms_incl_stt": summarize(end_to_end_latencies),
        "per_stage_latency_ms": {stage: summarize(v) for stage, v in stage_latencies.items()},
        "under_200ms_target": {
            "local_pipeline_p50_under_200ms": summarize(local_pipeline_latencies)["p50"] < 200,
            "local_pipeline_p70_under_200ms": summarize(local_pipeline_latencies)["p70"] < 200,
            "local_pipeline_p100_under_200ms": summarize(local_pipeline_latencies)["p100_max"] < 200,
        },
    }
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report, records
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of sampled queries (cycled from the query set)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=str, default="benchmark_report.json")
    args = parser.parse_args()
    report, records = run_benchmark(n_queries=args.n, top_k=args.top_k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "records": records}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full report to {args.out}")