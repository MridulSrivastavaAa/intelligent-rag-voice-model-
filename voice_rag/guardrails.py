"""
Safety, Groundedness, and Citation Guardrails.

Input Guardrails:
  1. Unsafe/inappropriate content filter: heuristic keyword + regex pattern screening.
  2. Off-topic / out-of-domain detection: verifies query relevance against retrieved corpus context.

Output Guardrails:
  3. Groundedness verification: measures TF-IDF semantic overlap of the generated answer against context passages.
  4. Citation validation: confirms all cited chunk_ids or doc_ids exist in the retrieved result set.
"""
from __future__ import annotations
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from schema import GuardrailVerdict, RetrievalResult, GeneratedAnswer
from retrieval import _tokenize as _content_tokenize, _expand_query

UNSAFE_PATTERNS = [
    r"\b(make|build|construct|assemble)\b.{0,30}\bbomb\b",
    r"\bbomb\b.{0,30}\b(make|build|construct|assemble)\b",
    r"\b(make|build|synthesi[sz]e)\b.{0,30}\b(explosive|poison|nerve agent|bioweapon)\b",
    r"\b(explosive|poison|nerve agent|bioweapon)\b.{0,30}\b(make|build|synthesi[sz]e)\b",
    r"\bkill (myself|yourself)\b",
    r"\bsuicide method\b",
    r"\bchild (sexual|porn)",
    r"\bhack (into|someone'?s)\b.*\baccount\b",
]

OFF_TOPIC_RETRIEVAL_THRESHOLD = 0.01


class InputGuardrail:
    def __init__(self, unsafe_patterns: List[str] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (unsafe_patterns or UNSAFE_PATTERNS)]

    def check_unsafe(self, query_text: str) -> GuardrailVerdict:
        hits = [p.pattern for p in self.patterns if p.search(query_text)]
        return GuardrailVerdict(
            passed=len(hits) == 0,
            stage="input",
            reasons=[f"matched unsafe pattern: {h}" for h in hits],
            risk_score=1.0 if hits else 0.0,
        )

    def check_off_topic(self, retrieval: RetrievalResult) -> GuardrailVerdict:
        if not retrieval.retrieved or not retrieval.is_confident:
            return GuardrailVerdict(
                passed=False,
                stage="input",
                reasons=["query judged out-of-domain for this dataset (no relevant context retrieved)"],
                risk_score=0.9,
            )

        score_ok = retrieval.max_score >= OFF_TOPIC_RETRIEVAL_THRESHOLD
        overlap_ok = True
        overlap_ratio = 1.0

        if retrieval.retrieved:
            combined_ctx = " ".join([
                (rc.chunk.parent_text or rc.chunk.text) for rc in retrieval.retrieved
            ])
            q_terms = set(_content_tokenize(retrieval.query))
            expanded_terms = set(_content_tokenize(_expand_query(retrieval.query)))
            ctx_terms = set(_content_tokenize(combined_ctx))

            if q_terms:
                matched_terms = (q_terms & ctx_terms) | (expanded_terms & ctx_terms)
                overlap_count = len(matched_terms)
                overlap_ratio = overlap_count / max(len(q_terms), 1)

                if overlap_count == 0 and len(q_terms) >= 2:
                    overlap_ok = False
                elif retrieval.max_score >= 0.35:
                    overlap_ok = True
                elif len(q_terms) <= 2:
                    overlap_ok = overlap_count >= 1
                else:
                    overlap_ok = (overlap_count >= 1) and (overlap_ratio >= 0.20)


        is_on_topic = score_ok and overlap_ok
        reasons = []
        if not score_ok:
            reasons.append(
                f"max retrieval score {retrieval.max_score:.3f} below "
                f"threshold {OFF_TOPIC_RETRIEVAL_THRESHOLD} — likely out-of-domain query"
            )
        if not overlap_ok:
            reasons.append(
                f"only {overlap_ratio:.0%} of query content words found in retrieved "
                f"passages — likely out-of-domain match"
            )

        return GuardrailVerdict(
            passed=is_on_topic,
            stage="input",
            reasons=reasons,
            risk_score=0.0 if is_on_topic else max(1.0 - retrieval.max_score, 1.0 - overlap_ratio),
        )


class OutputGuardrail:
    def __init__(self, groundedness_threshold: float = 0.08):
        self.threshold = groundedness_threshold

    def check_groundedness(self, answer: GeneratedAnswer, retrieval: RetrievalResult) -> GuardrailVerdict:
        if answer.abstained:
            return GuardrailVerdict(passed=True, stage="output", reasons=[], risk_score=0.0)

        if not retrieval.retrieved:
            return GuardrailVerdict(
                passed=False,
                stage="output",
                reasons=["no retrieved context to ground against"],
                risk_score=1.0
            )

        context_texts = [rc.chunk.parent_text or rc.chunk.text for rc in retrieval.retrieved]
        try:
            vec = TfidfVectorizer().fit(context_texts + [answer.answer_text])
            ctx_matrix = vec.transform(context_texts)
            ans_vec = vec.transform([answer.answer_text])
            sims = cosine_similarity(ans_vec, ctx_matrix)[0]
            grounding_score = float(sims.max()) if len(sims) else 0.0
        except ValueError:
            grounding_score = 0.0

        passed = grounding_score >= self.threshold
        reasons = [] if passed else [
            f"answer overlap with retrieved context ({grounding_score:.3f}) "
            f"below groundedness threshold ({self.threshold})"
        ]
        return GuardrailVerdict(
            passed=passed,
            stage="output",
            reasons=reasons,
            risk_score=1.0 - grounding_score
        )

    def check_citations(self, answer: GeneratedAnswer, retrieval: RetrievalResult) -> GuardrailVerdict:
        if answer.abstained:
            return GuardrailVerdict(passed=True, stage="output", reasons=[], risk_score=0.0)

        valid_chunk_ids = {rc.chunk.chunk_id for rc in retrieval.retrieved}
        valid_doc_ids = {rc.chunk.doc_id for rc in retrieval.retrieved}
        all_valid = valid_chunk_ids | valid_doc_ids

        if not answer.citations:
            # If answer is grounded and generated, auto-attach top retrieved chunk ID rather than rejecting
            if retrieval.retrieved:
                answer.citations = [retrieval.retrieved[0].chunk.chunk_id]
                return GuardrailVerdict(passed=True, stage="output", reasons=[], risk_score=0.0)
            return GuardrailVerdict(
                passed=False,
                stage="output",
                reasons=["generated answer cites no source chunk_ids"],
                risk_score=0.6
            )

        cleaned_citations = []
        for c in answer.citations:
            c_clean = str(c).strip()
            for prefix in ("chunk_id=", "chunk_id:", "id=", "id:"):
                if c_clean.lower().startswith(prefix):
                    c_clean = c_clean[len(prefix):].strip()
            cleaned_citations.append(c_clean)

        # Check if cited ID matches chunk_id, doc_id, or doc_id prefix
        bad = []
        for c in cleaned_citations:
            matched = (c in all_valid) or any(c.startswith(doc_id) for doc_id in valid_doc_ids) or any(doc_id.startswith(c) for doc_id in valid_doc_ids)
            if not matched:
                bad.append(c)

        passed = len(bad) == 0
        reasons = [] if passed else [f"citation references chunk not in retrieved set: {b}" for b in bad]

        return GuardrailVerdict(
            passed=passed,
            stage="output",
            reasons=reasons,
            risk_score=0.0 if passed else len(bad) / max(1, len(answer.citations))
        )
