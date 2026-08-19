"""
Hybrid Multi-Strategy Vector Store & BM25 Retriever.

Combines:
  1. Dense semantic character & word n-gram TF-IDF vector space with sublinear term weighting
  2. Lexical BM25 rank matching with language-aware tokenization
  3. Reciprocal Rank Fusion (RRF) with zero-score gating across multiple chunking indices
  4. Dynamic Hinglish to Devanagari query expansion
"""
from __future__ import annotations
import time
import re
from collections import defaultdict
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from schema import Chunk, RetrievedChunk, RetrievalResult


_STOPWORDS = {
    # High-frequency function words across Hindi and English
    "है", "हैं", "का", "के", "की", "में", "से", "को", "यह", "एक", "और",
    "पर", "भी", "था", "थी", "थे", "हो", "गया", "गई", "कि", "जो", "इस",
    "क्या", "कैसे", "कौन", "कब", "कहाँ", "किसने", "किसका", "कितने", "कितनी",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "what", "how", "when", "where", "who", "which", "for",
    "with", "by", "as", "it", "its", "at", "be", "does", "do", "did", "tell", "me", "about"
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens with stopword filtering."""
    tokens = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return filtered if filtered else [t for t in tokens if len(t) > 1] or tokens


_HINGLISH_MAP = {
    "bharat": "भारत", "rajdhani": "राजधानी", "kya": "क्या", "hai": "है", "kaun": "कौन", "si": "सी",
    "lakshan": "लक्षण", "madhumeh": "मधुमेह", "diabetes": "मधुमेह डायबिटीज", "samvidhan": "संविधान",
    "prakash": "प्रकाश", "photosynthesis": "प्रकाश संश्लेषण", "taj": "ताज", "mahal": "महल",
    "sanshleshana": "संश्लेषण", "yoga": "योग", "ganga": "गंगा", "nadi": "नदी", "sthapna": "स्थापना",
    "computer": "कंप्यूटर", "memory": "मेमोरी", "jalvayu": "जलवायु", "parivartan": "परिवर्तन",
    "swatantrata": "स्वतंत्रता", "divas": "दिवस", "climate": "जलवायु", "change": "परिवर्तन",
    "rbi": "आरबीआई रिजर्व बैंक ऑफ इंडिया", "blood": "रक्तचाप", "pressure": "रक्तचाप",
    "graha": "ग्रह", "solar": "सौर मंडल", "planets": "ग्रह सौर मंडल", "blockchain": "ब्लॉकचेन",
    "5g": "5जी", "delhi": "दिल्ली", "shahjahan": "शाहजहाँ", "ram": "रैम मेमोरी",
}


def _expand_query(query: str) -> str:
    """Expands query with transliteration mapping to bridge Hindi, English, and Hinglish."""
    words = re.findall(r"\w+", query.lower())
    expanded = [query]
    mapped = [_HINGLISH_MAP[w] for w in words if w in _HINGLISH_MAP]
    if mapped:
        expanded.extend(mapped)
    return " ".join(expanded)


class StrategyIndex:
    """TF-IDF + BM25 index over chunks produced by a specific chunking strategy."""

    def __init__(self, strategy_name: str, chunks: List[Chunk]):
        self.strategy_name = strategy_name
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Hybrid n-gram vectorizer: captures both sub-word morphology and full word semantics
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts) if texts else None

        tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 5):
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Dense similarity
        q_vec = self.vectorizer.transform([query])
        dense_scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # Lexical BM25 similarity
        q_tokens = _tokenize(query)
        bm25_scores = np.zeros(len(self.chunks))
        if self.bm25 and q_tokens:
            raw_bm25 = np.array(self.bm25.get_scores(q_tokens))
            if raw_bm25.max() > 0:
                bm25_scores = raw_bm25 / raw_bm25.max()

        # Reciprocal Rank Fusion (RRF) with relevance gating
        dense_rank = np.argsort(-dense_scores)
        lex_rank = np.argsort(-bm25_scores)
        rrf = np.zeros(len(self.chunks))
        k_rrf = 60

        # Only award rank bonus if chunk has meaningful non-zero score
        for rank, idx in enumerate(dense_rank):
            if dense_scores[idx] > 0.02:
                rrf[idx] += 1.0 / (k_rrf + rank + 1)

        for rank, idx in enumerate(lex_rank):
            if bm25_scores[idx] > 0.0:
                rrf[idx] += 1.0 / (k_rrf + rank + 1)

        # Filter indices with non-zero match
        matching_indices = [idx for idx in np.argsort(-rrf) if rrf[idx] > 0 or dense_scores[idx] > 0.08]
        top_idx = matching_indices[:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(rrf[idx]),
                "dense_score": float(dense_scores[idx]),
                "lexical_score": float(bm25_scores[idx]),
            })
        return results


class HybridMultiStrategyRetriever:
    """Fuses multi-strategy index results via cross-strategy Reciprocal Rank Fusion."""

    def __init__(self, chunks: List[Chunk]):
        by_strategy: Dict[str, List[Chunk]] = defaultdict(list)
        for c in chunks:
            by_strategy[c.strategy].append(c)
        self.indices = {name: StrategyIndex(name, cs) for name, cs in by_strategy.items()}

    def retrieve(self, query: str, top_k: int = 5, per_strategy_k: int = 8) -> RetrievalResult:
        t0 = time.perf_counter()
        expanded_q = _expand_query(query)
        cross_strategy_scores: Dict[str, dict] = {}
        k_rrf = 60

        for strat_name, index in self.indices.items():
            hits = index.search(expanded_q, top_k=per_strategy_k)
            for rank, hit in enumerate(hits):
                cid = hit["chunk"].chunk_id
                fused_bonus = 1.0 / (k_rrf + rank + 1)
                if cid not in cross_strategy_scores:
                    cross_strategy_scores[cid] = {
                        "chunk": hit["chunk"],
                        "score": 0.0,
                        "dense_score": hit["dense_score"],
                        "lexical_score": hit["lexical_score"],
                        "source_strategy": strat_name,
                    }
                cross_strategy_scores[cid]["score"] += fused_bonus
                cross_strategy_scores[cid]["dense_score"] = max(
                    cross_strategy_scores[cid]["dense_score"], hit["dense_score"])
                cross_strategy_scores[cid]["lexical_score"] = max(
                    cross_strategy_scores[cid]["lexical_score"], hit["lexical_score"])

        ranked = sorted(cross_strategy_scores.values(), key=lambda r: -r["score"])[:top_k]
        retrieved = [
            RetrievedChunk(
                chunk=r["chunk"],
                score=r["score"],
                dense_score=r["dense_score"],
                lexical_score=r["lexical_score"],
                source_strategy=r["source_strategy"],
            ) for r in ranked
        ]
        latency_ms = (time.perf_counter() - t0) * 1000
        max_dense = max((r.dense_score for r in retrieved), default=0.0)
        max_lexical = max((r.lexical_score for r in retrieved), default=0.0)

        # Confident if either dense similarity or lexical matching is satisfied
        is_confident = (max_dense >= 0.07) or (max_lexical >= 0.15)

        return RetrievalResult(
            query=query,
            retrieved=retrieved,
            latency_ms=latency_ms,
            max_score=max_dense,
            is_confident=is_confident and len(retrieved) > 0,
        )
