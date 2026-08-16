"""
Chunking strategies for the RAG index.

We deliberately implement several distinct strategies rather than one
fixed-size splitter, because MSMARCO-XI passages vary a lot in length and
structure (short factoid passages vs. longer explanatory ones, Hindi vs.
English tokenization behaviour, etc.). Each strategy is registered and can
be run independently or combined; retrieval.py queries across all of them
and fuses the results.

Strategies:
1. FixedSizeChunker      - naive baseline, char-based windows with overlap.
2. SentenceWindowChunker - groups N sentences with stride overlap; better
                            respects semantic units than raw char windows.
3. SemanticChunker        - splits on sentence boundaries, then merges
                            adjacent sentences into a chunk until the
                            TF-IDF cosine similarity to the running chunk
                            drops below a threshold ("semantic breakpoint"),
                            bounded by min/max size. No embedding model
                            download required.
4. MetadataAwareChunker  - wraps any base chunker and does parent/child
                            chunking: small chunks are what's indexed and
                            matched against the query, but each carries a
                            pointer to the full parent passage, which is
                            what actually gets sent to the generator. Also
                            attaches source metadata (doc_id, language,
                            position) used later for filtering/guardrails.
"""
from __future__ import annotations
import re
from typing import List
from schema import Chunk

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # handles Hindi danda (।) too


def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text.strip()]


class BaseChunker:
    name = "base"

    def chunk(self, doc_id: str, text: str, language: str = "unknown") -> List[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """Naive baseline: fixed character window with overlap. Kept as a
    control/comparison strategy, not the primary retrieval strategy."""

    name = "fixed_size"

    def __init__(self, size: int = 220, overlap: int = 40):
        self.size = size
        self.overlap = overlap

    def chunk(self, doc_id, text, language="unknown"):
        chunks = []
        step = max(1, self.size - self.overlap)
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + self.size]
            if not piece.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{self.name}_{i}",
                doc_id=doc_id, text=piece.strip(), strategy=self.name,
                language=language, position=i,
            ))
            if start + self.size >= len(text):
                break
        return chunks


class SentenceWindowChunker(BaseChunker):
    """Groups `window` sentences per chunk, sliding by `stride` sentences,
    so consecutive chunks overlap and a fact split across a sentence
    boundary is still fully contained in at least one chunk."""

    name = "sentence_window"

    def __init__(self, window: int = 3, stride: int = 2):
        self.window = window
        self.stride = stride

    def chunk(self, doc_id, text, language="unknown"):
        sents = split_sentences(text)
        chunks = []
        i = 0
        pos = 0
        while i < len(sents):
            window_sents = sents[i:i + self.window]
            piece = " ".join(window_sents).strip()
            if piece:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{self.name}_{pos}",
                    doc_id=doc_id, text=piece, strategy=self.name,
                    language=language, position=pos,
                ))
                pos += 1
            if i + self.window >= len(sents):
                break
            i += self.stride
        return chunks


class SemanticChunker(BaseChunker):
    """Sentence-boundary + TF-IDF similarity breakpoint chunking.

    Walks sentence by sentence, growing the current chunk. Before adding
    the next sentence, checks its TF-IDF cosine similarity against the
    chunk built so far; if similarity drops below `threshold` (a topic
    shift) AND the chunk already meets `min_sents`, we close the chunk and
    start a new one. This approximates semantic chunking without needing
    a downloaded embedding model.
    """

    name = "semantic"

    def __init__(self, threshold: float = 0.12, min_sents: int = 1, max_sents: int = 6):
        self.threshold = threshold
        self.min_sents = min_sents
        self.max_sents = max_sents

    def chunk(self, doc_id, text, language="unknown"):
        sents = split_sentences(text)
        if not _SKLEARN_OK or len(sents) <= 2:
            # fall back to sentence-window behaviour for very short docs,
            # but keep this chunker's own name/chunk_id namespace so these
            # chunks don't collide with (and get double-counted against)
            # the real sentence_window strategy's chunks during
            # cross-strategy RRF fusion.
            fallback_chunks = SentenceWindowChunker(window=3, stride=3).chunk(doc_id, text, language)
            for c in fallback_chunks:
                c.chunk_id = c.chunk_id.replace("_sentence_window_", f"_{self.name}_")
                c.strategy = self.name
            return fallback_chunks

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(sents)
        sent_vecs = vec.transform(sents)

        chunks = []
        current = [0]
        pos = 0
        for idx in range(1, len(sents)):
            running_vec = vec.transform([" ".join(sents[s] for s in current)])
            sim = cosine_similarity(running_vec, sent_vecs[idx])[0][0]
            if sim < self.threshold and len(current) >= self.min_sents:
                piece = " ".join(sents[s] for s in current).strip()
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{self.name}_{pos}",
                    doc_id=doc_id, text=piece, strategy=self.name,
                    language=language, position=pos,
                ))
                pos += 1
                current = [idx]
            else:
                current.append(idx)
                if len(current) >= self.max_sents:
                    piece = " ".join(sents[s] for s in current).strip()
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_{self.name}_{pos}",
                        doc_id=doc_id, text=piece, strategy=self.name,
                        language=language, position=pos,
                    ))
                    pos += 1
                    current = []
        if current:
            piece = " ".join(sents[s] for s in current).strip()
            if piece:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{self.name}_{pos}",
                    doc_id=doc_id, text=piece, strategy=self.name,
                    language=language, position=pos,
                ))
        return chunks


class MetadataAwareChunker(BaseChunker):
    """Wraps a base chunker to do parent/child chunking + rich metadata.

    Retrieval matches happen against the (small) child chunk text, but each
    chunk carries `parent_text` (the full original passage) so the
    generator gets full context instead of a truncated fragment — this is
    the standard "small-to-big" retrieval pattern.
    """

    name = "metadata_aware"

    def __init__(self, base: BaseChunker):
        self.base = base

    def chunk(self, doc_id, text, language="unknown", extra_metadata: dict | None = None):
        base_chunks = self.base.chunk(doc_id, text, language)
        out = []
        for c in base_chunks:
            c.strategy = f"{self.name}::{self.base.name}"
            c.parent_text = text
            c.metadata = {
                "char_len": len(c.text),
                "n_sentences": len(split_sentences(c.text)),
                **(extra_metadata or {}),
            }
            out.append(c)
        return out


DEFAULT_STRATEGIES = {
    "fixed_size": FixedSizeChunker(size=220, overlap=40),
    "sentence_window": SentenceWindowChunker(window=3, stride=2),
    "semantic": SemanticChunker(threshold=0.12),
}


def build_all_chunks(docs: List[dict]) -> List[Chunk]:
    """docs: list of {id, text, language, metadata}. Runs every strategy,
    each wrapped for parent/child + metadata, and returns the union — the
    retriever later fuses matches across strategies."""
    all_chunks: List[Chunk] = []
    for d in docs:
        for strat in DEFAULT_STRATEGIES.values():
            wrapped = MetadataAwareChunker(strat)
            all_chunks.extend(
                wrapped.chunk(d["id"], d["text"], d.get("language", "unknown"),
                              extra_metadata=d.get("metadata", {}))
            )
    return all_chunks
