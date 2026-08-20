"""
Multi-Provider Answer Generation Engine.

Supported Backends:
  1. Groq (GroqGenerator) - Ultra-fast cloud inference for Llama 3.3 70B & 3.1 8B (via GROQ_API_KEY)
  2. Google Gemini (GeminiGenerator) - Gemini 2.0 Flash / 1.5 Flash (via GEMINI_API_KEY or GOOGLE_API_KEY)
  3. OpenAI (OpenAIGenerator) - GPT-4o-mini / GPT-4o (via OPENAI_API_KEY)
  4. Anthropic Claude (ClaudeGenerator) - Claude 3.5 Sonnet / Haiku (via ANTHROPIC_API_KEY)
  5. OpenRouter (OpenRouterGenerator) - OpenRouter Models (via OPENROUTER_API_KEY)
  6. Ollama (OllamaGenerator) - Local offline inference / HuggingFace GGUF models (via OLLAMA_HOST)
  7. Smart Semantic Extractive QA (SmartExtractiveGenerator / ExtractiveFallbackGenerator) -
     Deterministic offline answer synthesis with sentence-level scoring and grounded citations.
"""
from __future__ import annotations
import json
import os
import re
import time
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

from schema import RetrievalResult, GeneratedAnswer
from chunking import split_sentences

SYSTEM_PROMPT = (
    "You are an expert grounded question-answering assistant. You MUST answer accurately "
    "using ONLY the provided context chunks. Answer in the EXACT SAME LANGUAGE as the user's question "
    "(if Hindi question, answer in Hindi; if English question, answer in English). "
    "Be concise, clear, and direct. If the context does not contain the answer, set abstain=true. "
    "Always cite the chunk_id(s) you used. Respond ONLY with a valid JSON object of the form: "
    '{"answer": "...", "citations": ["chunk_id", ...], "abstain": false} '
    "with no other text and no markdown backticks."
)


def clean_citation_id(c: Any) -> str:
    c_clean = str(c).strip()
    for prefix in ("chunk_id=", "chunk_id:", "id=", "id:"):
        if c_clean.lower().startswith(prefix):
            c_clean = c_clean[len(prefix):].strip()
    return c_clean


def _parse_llm_json_response(raw_text: str) -> dict:
    """Safely extracts and parses JSON payload from LLM responses."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try searching for first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass

    return {"answer": raw_text.strip(), "citations": [], "abstain": False}


def _normalize_token(t: str) -> str:
    """Normalize Hindi and English inflections for matching."""
    t = t.lower().strip()
    for suf in ("ों", "े", "ी", "ीय", "िक", "ायें", "एं", "ियों"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[:-len(suf)]
            break
    for suf in ("ing", "tion", "tions", "s", "es", "ed", "ial", "al"):
        if t.endswith(suf) and len(t) > len(suf) + 3:
            t = t[:-len(suf)]
            break
    return t


def _term_in_text(term: str, text: str, word_set: set[str]) -> bool:
    t_norm = _normalize_token(term)
    if term in text or t_norm in text:
        return True
    for w in word_set:
        w_norm = _normalize_token(w)
        if (t_norm in w_norm) or (w_norm in t_norm) or (len(t_norm) >= 4 and t_norm[:4] == w_norm[:4]):
            return True
    return False


class SmartExtractiveGenerator:
    """Intelligent semantic sentence-scoring extractive QA engine.
    Used when no cloud API keys are present or when offline.
    Scores individual sentences against the question to synthesize a concise, direct answer."""

    def __init__(self):
        pass

    def generate(self, query: str, retrieval: RetrievalResult) -> GeneratedAnswer:
        t0 = time.perf_counter()

        if not retrieval.retrieved or not retrieval.is_confident:
            latency_ms = (time.perf_counter() - t0) * 1000
            return GeneratedAnswer(
                answer_text=(
                    "I don't have enough grounded information in the dataset to answer that question confidently."
                ),
                citations=[],
                grounded=False,
                grounding_score=0.0,
                abstained=True,
                latency_ms=latency_ms,
            )

        # Extract words from query
        q_clean = query.lower()
        q_tokens = set(re.findall(r"[\w\u0900-\u097F]+", q_clean))
        from retrieval import _STOPWORDS
        q_content_tokens = {t for t in q_tokens if t not in _STOPWORDS and len(t) > 1}
        q_norm_tokens = {_normalize_token(t) for t in q_content_tokens}

        # Check for ungrounded / missing distinctive entity terms in query
        all_retrieved_text = " ".join([
            (rc.chunk.parent_text or rc.chunk.text).lower() for rc in retrieval.retrieved
        ])
        all_retrieved_words = set(re.findall(r"[\w\u0900-\u097F]+", all_retrieved_text))

        distinct_missing = [
            t for t in q_content_tokens
            if len(t) >= 4 and not _term_in_text(t, all_retrieved_text, all_retrieved_words)
        ]

        if distinct_missing and len(distinct_missing) >= 1 and retrieval.max_score < 0.50:
            latency_ms = (time.perf_counter() - t0) * 1000
            return GeneratedAnswer(
                answer_text=(
                    "I don't have enough grounded information in the dataset to answer that question accurately."
                ),
                citations=[],
                grounded=False,
                grounding_score=0.0,
                abstained=True,
                latency_ms=latency_ms,
            )

        # Gather sentences across top retrieved chunks
        scored_sentences = []

        for rc in retrieval.retrieved[:3]:
            chunk_id = rc.chunk.chunk_id
            text = rc.chunk.parent_text or rc.chunk.text
            sents = split_sentences(text)

            for idx, sent in enumerate(sents):
                sent_clean = sent.lower()
                sent_tokens = set(re.findall(r"[\w\u0900-\u097F]+", sent_clean))
                sent_norm = {_normalize_token(t) for t in sent_tokens}

                # Overlap score with query content tokens
                overlap = sum(1 for qt in q_content_tokens if _term_in_text(qt, sent_clean, sent_tokens))
                score = (overlap * 4.0) + (rc.dense_score * 2.0) + (rc.lexical_score * 1.5)

                # Positional preference for earlier summary sentences
                if idx == 0:
                    score += 0.5
                elif idx == 1:
                    score += 0.2

                # Length penalty for tiny fragments
                if len(sent) < 15:
                    score -= 1.0

                scored_sentences.append({
                    "sentence": sent.strip(),
                    "score": score,
                    "chunk_id": chunk_id,
                    "overlap": overlap,
                    "order": idx
                })

        # Sort by relevance score
        scored_sentences.sort(key=lambda x: -x["score"])

        if scored_sentences and (scored_sentences[0]["overlap"] > 0 or retrieval.max_score >= 0.2):
            best = scored_sentences[0]
            selected = [best["sentence"]]
            cited_ids = {best["chunk_id"]}

            # If second sentence is strongly related and from same chunk/flow
            if len(scored_sentences) > 1 and scored_sentences[1]["score"] >= best["score"] * 0.75:
                second = scored_sentences[1]
                if second["sentence"] not in selected:
                    selected.append(second["sentence"])
                    cited_ids.add(second["chunk_id"])

            answer_text = " ".join(selected).strip()
            latency_ms = (time.perf_counter() - t0) * 1000

            return GeneratedAnswer(
                answer_text=answer_text,
                citations=list(cited_ids),
                grounded=True,
                grounding_score=retrieval.max_score,
                abstained=False,
                latency_ms=latency_ms,
            )

        # Fallback to top retrieved passage first 2 sentences if general match
        top = retrieval.retrieved[0]
        source_text = top.chunk.parent_text or top.chunk.text
        sents = split_sentences(source_text)
        answer_text = " ".join(sents[:2]).strip()
        citations = [top.chunk.chunk_id]
        latency_ms = (time.perf_counter() - t0) * 1000

        return GeneratedAnswer(
            answer_text=answer_text,
            citations=citations,
            grounded=True,
            grounding_score=top.dense_score,
            abstained=False,
            latency_ms=latency_ms,
        )


# Alias for backward compatibility
ExtractiveFallbackGenerator = SmartExtractiveGenerator


class BaseLLMGenerator:
    """Base class for API-driven LLM generators."""
    def __init__(self):
        self.fallback = SmartExtractiveGenerator()

    def _build_context(self, retrieval: RetrievalResult) -> str:
        blocks = []
        for rc in retrieval.retrieved:
            text = rc.chunk.parent_text or rc.chunk.text
            blocks.append(f"[chunk_id={rc.chunk.chunk_id}] {text}")
        return "\n\n".join(blocks)


class GroqGenerator(BaseLLMGenerator):
    """Cloud generation via Groq API (ultra-fast Llama 3.3 70B & 3.1 8B)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not self.api_key or not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)

        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()

        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _parse_llm_json_response(content)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]

                return GeneratedAnswer(
                    answer_text=parsed.get("answer", ""),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
            except Exception:
                time.sleep(0.2 * attempt)
                continue

        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb


class GeminiGenerator(BaseLLMGenerator):
    """Cloud generation via Google Gemini API (Gemini Flash / Pro)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        self.candidate_models = [
            self.model,
            "gemini-flash-latest",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-pro-latest",
            "gemini-2.5-flash",
        ]

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not self.api_key or not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)

        context = self._build_context(retrieval)
        prompt = f"{SYSTEM_PROMPT}\n\nContext chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()

        # Try active model and fallbacks if necessary
        models_to_try = []
        for m in self.candidate_models:
            if m not in models_to_try:
                models_to_try.append(m)

        last_err = None
        for current_model in models_to_try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self.api_key}"
            for attempt in range(1, max_retries + 2):
                try:
                    resp = requests.post(
                        endpoint,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key,
                        },
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.1,
                                "responseMimeType": "application/json",
                            }
                        },
                        timeout=15,
                    )
                    if resp.status_code == 404:
                        # Model retired/renamed, try next model in candidate list
                        break
                    resp.raise_for_status()
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        break
                    content = candidates[0]["content"]["parts"][0]["text"]
                    parsed = _parse_llm_json_response(content)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    raw_cits = parsed.get("citations", [])
                    cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]

                    # Remember working model
                    self.model = current_model

                    return GeneratedAnswer(
                        answer_text=parsed.get("answer", ""),
                        citations=cits,
                        grounded=not parsed.get("abstain", False),
                        abstained=bool(parsed.get("abstain", False)),
                        latency_ms=latency_ms,
                        attempts=attempt,
                    )
                except Exception as ex:
                    last_err = ex
                    time.sleep(0.2 * attempt)
                    continue

        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb



class OpenAIGenerator(BaseLLMGenerator):
    """Cloud generation via OpenAI API (GPT-4o-mini / GPT-4o)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not self.api_key or not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)

        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()

        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _parse_llm_json_response(content)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]

                return GeneratedAnswer(
                    answer_text=parsed.get("answer", ""),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
            except Exception:
                time.sleep(0.2 * attempt)
                continue

        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb


class ClaudeGenerator(BaseLLMGenerator):
    """Cloud generation via Anthropic Claude API."""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022"):
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
        self.endpoint = "https://api.anthropic.com/v1/messages"

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not self.api_key or not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)

        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()

        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 400,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                parsed = _parse_llm_json_response(text)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]

                return GeneratedAnswer(
                    answer_text=parsed.get("answer", ""),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
            except Exception:
                time.sleep(0.2 * attempt)
                continue

        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb


class OpenRouterGenerator(BaseLLMGenerator):
    """Cloud generation via OpenRouter API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not self.api_key or not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)

        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()

        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _parse_llm_json_response(content)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]

                return GeneratedAnswer(
                    answer_text=parsed.get("answer", ""),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
            except Exception:
                time.sleep(0.2 * attempt)
                continue

        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb


class OllamaGenerator(BaseLLMGenerator):
    """Offline/local generation via Ollama."""

    def __init__(self, host: str | None = None, model: str | None = None,
                 temperature: float = 0.1, request_timeout: int = 30):
        super().__init__()
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.temperature = temperature
        self.timeout = request_timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=1.5)
            return r.ok
        except Exception:
            return False

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 1) -> GeneratedAnswer:
        if not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)

        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()

        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "format": "json",
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                parsed = _parse_llm_json_response(text)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]
                if not cits and retrieval.retrieved:
                    cits = [retrieval.retrieved[0].chunk.chunk_id]

                return GeneratedAnswer(
                    answer_text=parsed.get("answer", "").strip(),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
            except Exception:
                time.sleep(0.2 * attempt)
                continue

        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb


def make_generator(backend: str | None = None):
    """Factory creating the best available generator with zero-configuration auto-detection.

    Order of priority:
      1. Explicitly requested backend
      2. Groq (if GROQ_API_KEY present)
      3. Google Gemini (if GEMINI_API_KEY / GOOGLE_API_KEY present)
      4. OpenAI (if OPENAI_API_KEY present)
      5. Claude (if ANTHROPIC_API_KEY present)
      6. OpenRouter (if OPENROUTER_API_KEY present)
      7. Ollama (if running and reachable at OLLAMA_HOST)
      8. Smart Extractive Synthesizer (always available)
    """
    raw_backend = (backend or os.environ.get("GENERATOR_BACKEND", "auto")).lower()

    if raw_backend == "groq":
        return GroqGenerator()
    if raw_backend in ("gemini", "google"):
        return GeminiGenerator()
    if raw_backend == "openai":
        return OpenAIGenerator()
    if raw_backend in ("claude", "anthropic"):
        return ClaudeGenerator()
    if raw_backend == "openrouter":
        return OpenRouterGenerator()
    if raw_backend == "ollama":
        gen = OllamaGenerator()
        if gen.is_available():
            return gen
        return SmartExtractiveGenerator()
    if raw_backend in ("extractive", "fallback", "offline"):
        return SmartExtractiveGenerator()

    # Auto-detection mode (default) - Google Gemini prioritized
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiGenerator()
    if os.environ.get("GROQ_API_KEY"):
        return GroqGenerator()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerator()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeGenerator()
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterGenerator()

    # Check local Ollama
    ollama_gen = OllamaGenerator()
    if ollama_gen.is_available():
        return ollama_gen

    # High-accuracy smart extractive fallback
    return SmartExtractiveGenerator()

