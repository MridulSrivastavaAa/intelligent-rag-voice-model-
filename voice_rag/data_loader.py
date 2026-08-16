"""
Dataset loading.

Real path (`load_msmarco_xi`): uses the `datasets` library to pull
ai4bharat/MSMARCO-XI from the HuggingFace Hub. This is the code that
should be used in any environment with normal internet access.

This sandbox's network egress is restricted to package registries
(pypi/npm/github/crates/etc.) and does not include huggingface.co, so the
real download will fail here with a connection error. `load_docs()`
detects that and transparently falls back to the bundled
`data/sample_msmarco_xi.jsonl`, which mirrors the same
query/passage/language schema on a small hand-curated set, so the rest of
the pipeline (chunking/retrieval/guardrails/benchmark) is fully
exercisable offline.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_msmarco_xi.jsonl")


def load_msmarco_xi(split: str = "train", language: str = "hi", limit: int | None = 500):
    """Real loader — requires network access to huggingface.co."""
    from datasets import load_dataset  # imported lazily; optional dependency
    ds = load_dataset("ai4bharat/MSMARCO-XI", language, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_docs(limit: int | None = None) -> List[Dict]:
    """Returns a list of {id, text, language, metadata} dicts ready for
    chunking.build_all_chunks(). Tries the real HF dataset first, falls
    back to the bundled sample on any failure (no network, missing
    `datasets` package, gated dataset, etc.)."""
    try:
        ds = load_msmarco_xi(limit=limit or 500)
        docs = []
        for i, row in enumerate(ds):
            docs.append({
                "id": f"hf_{i}",
                "text": row.get("passage") or row.get("text") or "",
                "language": row.get("language", "hi"),
                "metadata": {"query": row.get("query", ""), "source": "ai4bharat/MSMARCO-XI"},
            })
        return docs
    except Exception as e:  # noqa: BLE001
        print(f"[data_loader] Falling back to bundled sample dataset "
              f"(could not reach the real MSMARCO-XI dataset: {e})")
        return load_sample_docs(limit=limit)


def load_sample_docs(limit: int | None = None) -> List[Dict]:
    docs = []
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append({
                "id": row["id"],
                "text": row["passage"],
                "language": row["language"],
                "metadata": {"query": row["query"], "source": "sample_msmarco_xi"},
            })
    return docs[:limit] if limit else docs


def load_sample_queries(limit: int | None = None) -> List[Dict]:
    """The `query` field of each sample row doubles as a realistic
    benchmark query set (that's literally what MS MARCO query/passage
    pairs are for)."""
    queries = []
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries.append({"query": row["query"], "expected_doc_id": row["id"], "language": row["language"]})
    return queries[:limit] if limit else queries
