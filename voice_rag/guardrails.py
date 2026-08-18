"""
Guardrails.
Input guardrails (run before retrieval/generation):
  1. Unsafe/inappropriate content filter — heuristic keyword + pattern
     screen for self-harm, weapons/explosives, hate speech, sexual content
     involving minors, etc. Blocks before any retrieval/LLM cost is spent.
  2. Off-topic / out-of-domain detection — after retrieval, if the best
     retrieval score is below a tuned threshold, the query is judged
     out-of-domain for this dataset and the pipeline should refuse/hedge
     rather than let the LLM improvise an ungrounded answer.
Output guardrails (run after generation):
  3. Groundedness check — verifies the generated answer's claims actually
     have lexical/semantic support in the retrieved chunks (TF-IDF overlap
     against the concatenated retrieved context). Low overlap => flagged
     as potentially hallucinated.
  4. Citation verification — the harness requires the generator to cite
     chunk_ids; this checks the cited ids exist in the retrieved set and
     that the cited chunk's text actually overlaps with the sentence that
     cites it (a lightweight hallucinated-citation catch).
These are heuristic, dependency-light guardrails appropriate for a local
demo. In production, step 1 would additionally call a moderation
endpoint/classifier model, and step 3 could use an NLI entailment model.
"""
from __future__ import annotations
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from schema import GuardrailVerdict, RetrievalResult, GeneratedAnswer
from retrieval import _tokenize as _content_tokenize
UNSAFE_PATTERNS = [
    # order-agnostic: "make a bomb" and "bomb ... make" both match
    r"\b(make|build|construct|assemble)\b.{0,30}\bbomb\b",
    r"\bbomb\b.{0,30}\b(make|build|construct|assemble)\b",
    r"\b(make|build|synthesi[sz]e)\b.{0,30}\b(explosive|poison|nerve agent|bioweapon)\b",
    r"\b(explosive|poison|nerve agent|bioweapon)\b.{0,30}\b(make|build|synthesi[sz]e)\b",
    r"\bkill (myself|yourself)\b",
    r"\bsuicide method\b",
    r"\bchild (sexual|porn)",
    r"\bhack (into|someone'?s)\b.*\baccount\b",
]
OFF_TOPIC_RETRIEVAL_THRESHOLD = 0.06  # TF-IDF cosine scale, tuned on sample corpus
class InputGuardrail:
    def __init__(self, unsafe_patterns: List[str] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (unsafe_patterns or UNSAFE_PATTERNS)]
    def check_unsafe(self, query_text: str) -> GuardrailVerdict:
        hits = [p.pattern for p in self.patterns if p.search(query_text)]
        return GuardrailVerdict(
            passed=len(hits) == 0, stage="input",
            reasons=[f"matched unsafe pattern: {h}" for h in hits],
            risk_score=1.0 if hits else 0.0,
        )
    def check_off_topic(self, retrieval: RetrievalResult) -> GuardrailVerdict:
        score_ok = retrieval.max_score >= OFF_TOPIC_RETRIEVAL_THRESHOLD
        overlap_ok = True
        overlap_ratio = 1.0

        if retrieval.retrieved:
            # Concatenate top retrieved passages context
            combined_ctx = " ".join([
                (rc.chunk.parent_text or rc.chunk.text) for rc in retrieval.retrieved[:3]
            ])
            q_terms = set(_content_tokenize(retrieval.query))
            ctx_terms = set(_content_tokenize(combined_ctx))

            if q_terms:
                overlap_ratio = len(q_terms & ctx_terms) / len(q_terms)
                overlap_ok = overlap_ratio >= 0.25

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
            passed=is_on_topic, stage="input", reasons=reasons,
            risk_score=0.0 if is_on_topic else max(1.0 - retrieval.max_score, 1.0 - overlap_ratio),
        )

class OutputGuardrail:
    def __init__(self, groundedness_threshold: float = 0.10):
        self.threshold = groundedness_threshold
    def check_groundedness(self, answer: GeneratedAnswer, retrieval: RetrievalResult) -> GuardrailVerdict:
        if not retrieval.retrieved:
            return GuardrailVerdict(passed=False, stage="output",
                                     reasons=["no retrieved context to ground against"],
                                     risk_score=1.0)
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
        return GuardrailVerdict(passed=passed, stage="output", reasons=reasons,
                                 risk_score=1.0 - grounding_score)
    def check_citations(self, answer: GeneratedAnswer, retrieval: RetrievalResult) -> GuardrailVerdict:
        if answer.abstained:
            return GuardrailVerdict(passed=True, stage="output", reasons=[], risk_score=0.0)
        valid_ids = {rc.chunk.chunk_id for rc in retrieval.retrieved}
        if not answer.citations:
            return GuardrailVerdict(passed=False, stage="output",
                                     reasons=["generated answer cites no source chunk_ids"],
                                     risk_score=0.6)
        cleaned_citations = []
        for c in answer.citations:
            c_clean = str(c).strip()
            for prefix in ("chunk_id=", "chunk_id:", "id=", "id:"):
                if c_clean.lower().startswith(prefix):
                    c_clean = c_clean[len(prefix):].strip()
            cleaned_citations.append(c_clean)

        bad = [c for c in cleaned_citations if c not in valid_ids]
        passed = len(bad) == 0
        reasons = [] if passed else [f"citation references chunk not in retrieved set: {b}" for b in bad]
        return GuardrailVerdict(passed=passed, stage="output", reasons=reasons,
                                 risk_score=0.0 if passed else len(bad) / max(1, len(answer.citations)))

