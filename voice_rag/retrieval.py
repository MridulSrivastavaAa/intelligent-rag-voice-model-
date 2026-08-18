"""
Vector store / retriever.

Design notes on the embedding choice: sentence-transformer models are
normally pulled from the HuggingFace Hub at first use. This sandbox's
network egress is restricted to package registries (pypi/npm/github) and
does not include huggingface.co, so a HF download would hang/fail here.
We therefore use a TF-IDF vector space (scikit-learn) as the "dense"
embedding stand-in — it is a real, legitimate offline embedding technique
and keeps the whole pipeline self-contained and fast. EmbeddingProvider is
a small interface so a production deployment can swap in
sentence-transformers, OpenAI/Voyage/Sarvam embeddings, etc. without
touching the retriever logic.

Retrieval itself is hybrid:
  - "dense" similarity  (TF-IDF cosine)
  - "lexical" similarity (BM25, rank_bm25)
fused via Reciprocal Rank Fusion (RRF), and run independently against
every chunking strategy's index, then fused again across strategies.
This is the "multiple chunking strategies + real retrieval sophistication"
requirement.
"""
from __future__ import annotations
import time
from collections import defaultdict
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from schema import Chunk, RetrievedChunk, RetrievalResult


_STOPWORDS = {
    # high-frequency function words that otherwise swamp TF-IDF/BM25
    # similarity between unrelated passages once content words are
    # stripped by inflection mismatches. Small hand-picked set covering
    # Hindi + English function words seen in the sample corpus.
    "है", "हैं", "का", "के", "की", "में", "से", "को", "यह", "एक", "और",
    "पर", "भी", "था", "थी", "थे", "हो", "गया", "गई", "कि", "जो", "इस",
    "क्या", "कैसे", "कौन", "कब", "कहाँ",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "what", "how", "when", "where", "who", "which", "for",
    "with", "by", "as", "it", "its", "at", "be",
}


def _tokenize(text: str) -> List[str]:
    # simple whitespace/punct tokenizer with stopword removal; good
    # enough for BM25 over both Hindi (Devanagari, whitespace-delimited)
    # and English text.
    import re
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class StrategyIndex:
    """One TF-IDF + BM25 index over the chunks produced by a single
    chunking strategy."""

    def __init__(self, strategy_name: str, chunks: List[Chunk]):
        self.strategy_name = strategy_name
        self.chunks = chunks
        texts = [c.text for c in chunks]
        # Character n-gram TF-IDF as the "dense" signal: robust to Hindi
        # (and English) morphological inflection — e.g. "लक्षण" vs
        # "लक्षणों" — that would otherwise defeat exact word-token
        # matching, without requiring a downloaded embedding model.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts) if texts else None
        tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 5):
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        dense_scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        q_tokens = _tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # Reciprocal Rank Fusion between dense and lexical rankings
        dense_rank = np.argsort(-dense_scores)
        lex_rank = np.argsort(-bm25_scores)
        rrf = np.zeros(len(self.chunks))
        k_rrf = 60
        for rank, idx in enumerate(dense_rank):
            rrf[idx] += 1.0 / (k_rrf + rank + 1)
        for rank, idx in enumerate(lex_rank):
            rrf[idx] += 1.0 / (k_rrf + rank + 1)

        top_idx = np.argsort(-rrf)[:top_k]
        results = []
        for idx in top_idx:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(rrf[idx]),
                "dense_score": float(dense_scores[idx]),
                "lexical_score": float(bm25_scores[idx]),
            })
        return results


_HINGLISH_MAP = {
    "bharat": "भारत", "rajdhani": "राजधानी", "kya": "क्या", "hai": "है", "kaun": "कौन", "si": "सी",
    "lakshan": "लक्षण", "madhumeh": "मधुमेह", "samvidhan": "संविधान", "prakash": "प्रकाश",
    "photosynthesis": "प्रकाश संश्लेषण", "taj": "ताज", "mahal": "महल", "sanshleshana": "संश्लेषण",
    "yoga": "योग", "ganga": "गंगा", "nadi": "नदी", "sthapna": "स्थापना", "computer": "कंप्यूटर",
    "memory": "मेमोरी", "jalvayu": "जलवायु", "parivartan": "परिवर्तन", "swatantrata": "स्वतंत्रता",
    "divas": "दिवस", "climate": "जलवायु", "change": "परिवर्तन", "rbi": "आरबीआई",
}


def _expand_query(query: str) -> str:
    import re
    words = re.findall(r"\w+", query.lower())
    expanded = [query]
    mapped = [_HINGLISH_MAP[w] for w in words if w in _HINGLISH_MAP]
    if mapped:
        expanded.append(" ".join(mapped))
    return " ".join(expanded)


class HybridMultiStrategyRetriever:
    """Builds one StrategyIndex per chunking strategy present in the chunk
    set, searches all of them, and fuses results across strategies (again
    via RRF) so no single chunking heuristic can dominate or blind-spot
    the answer."""

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

        ranked = sorted(cross_strategy_scores.values(), key=lambda r: -r["score"])[:top_k]
        retrieved = [
            RetrievedChunk(
                chunk=r["chunk"], score=r["score"], dense_score=r["dense_score"],
                lexical_score=r["lexical_score"], source_strategy=r["source_strategy"],
            ) for r in ranked
        ]
        latency_ms = (time.perf_counter() - t0) * 1000
        max_score = max((r.dense_score for r in retrieved), default=0.0)
        return RetrievalResult(
            query=query, retrieved=retrieved, latency_ms=latency_ms,
            max_score=max_score,
            is_confident=max_score >= 0.08,  # tuned against TF-IDF cosine scale
        )

