"""
Answer generation.
Primary path: **Ollama**, running a fully offline/local model — including
models pulled straight from the Hugging Face Hub in GGUF format via
Ollama's `hf.co/<repo>` support (e.g. `ollama pull
hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF`, or any other GGUF repo you
point it at). The model is prompted to answer strictly from the retrieved
chunks and to return structured JSON
{"answer": str, "citations": [chunk_id, ...], "abstain": bool} so the
harness can parse it deterministically and the output guardrails can
verify citations. Configure via `OLLAMA_HOST` (default
`http://localhost:11434`) and `OLLAMA_MODEL` env vars, or pass them to
`OllamaGenerator(...)` directly.
A `ClaudeGenerator` (Anthropic API) is also kept for reference/comparison
— see `make_generator()` to switch backends.
Fallback path (used whenever the configured backend is unreachable, no
model/host is configured, or the API call fails after retries): a
deterministic extractive generator that composes an answer directly from
the top retrieved chunk(s) and always attaches correct citations. This
keeps the full pipeline runnable end-to-end (and its retrieval-side
latency benchmark-able) without any external network/API/local-server
dependency, while preserving the same structured-output contract the
harness expects.
"""
from __future__ import annotations
import json
import os
import time
import requests
from dotenv import load_dotenv
load_dotenv()
from schema import RetrievalResult, GeneratedAnswer
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant. You MUST answer only "
    "using the provided context chunks. If the context does not contain the "
    "answer, set abstain=true and explain briefly why. Always cite the "
    "chunk_id(s) you used. Respond with ONLY a JSON object of the form: "
    '{"answer": "...", "citations": ["chunk_id", ...], "abstain": false} '
    "with no other text, no markdown fences."
)


def clean_citation_id(c: str) -> str:
    c_clean = str(c).strip()
    for prefix in ("chunk_id=", "chunk_id:", "id=", "id:"):
        if c_clean.lower().startswith(prefix):
            c_clean = c_clean[len(prefix):].strip()
    return c_clean


class ExtractiveFallbackGenerator:

    """No-API-key / offline path: builds an answer directly from the best
    retrieved chunk(s) rather than calling an LLM. Deterministic and
    always correctly cited."""
    def generate(self, query: str, retrieval: RetrievalResult) -> GeneratedAnswer:
        t0 = time.perf_counter()
        if not retrieval.retrieved or not retrieval.is_confident:
            latency_ms = (time.perf_counter() - t0) * 1000
            return GeneratedAnswer(
                answer_text=("I don't have enough grounded information in the provided "
                             "dataset to answer that confidently."),
                citations=[], grounded=False, grounding_score=0.0,
                abstained=True, latency_ms=latency_ms,
            )
        top = retrieval.retrieved[0]
        # extractive summary: first 2 sentences of the (parent) chunk text
        source_text = top.chunk.parent_text or top.chunk.text
        from chunking import split_sentences
        sents = split_sentences(source_text)
        answer_text = " ".join(sents[:2]).strip()
        citations = [rc.chunk.chunk_id for rc in retrieval.retrieved[:2] if rc.score > 0]
        latency_ms = (time.perf_counter() - t0) * 1000
        return GeneratedAnswer(
            answer_text=answer_text, citations=citations, grounded=True,
            grounding_score=top.dense_score, abstained=False, latency_ms=latency_ms,
        )
class OllamaGenerator:
    """Offline/local generation via Ollama.
    Works with any model Ollama has pulled — including GGUF models
    pulled directly from the Hugging Face Hub, e.g.::
        ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
        ollama pull hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF
    or any of Ollama's own library models (``ollama pull llama3.1``,
    ``ollama pull qwen2.5``, ``ollama pull phi4``, ...). Point
    ``OLLAMA_MODEL`` at whatever tag ``ollama list`` shows after pulling.
    Uses Ollama's ``/api/chat`` endpoint with ``format="json"`` so the
    server enforces valid JSON output matching our structured contract,
    same as the Claude path.
    """
    def __init__(self, host: str | None = None, model: str | None = None,
                 temperature: float = 0.1, request_timeout: int = 30):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.temperature = temperature
        self.timeout = request_timeout
        self.fallback = ExtractiveFallbackGenerator()
    def _build_context(self, retrieval: RetrievalResult) -> str:
        blocks = []
        for rc in retrieval.retrieved:
            text = rc.chunk.parent_text or rc.chunk.text
            blocks.append(f"[chunk_id={rc.chunk.chunk_id}] {text}")
        return "\n\n".join(blocks)
    def is_available(self) -> bool:
        """Quick health check — used by make_generator() to decide
        whether to wire up Ollama or go straight to the offline fallback,
        and handy for diagnostics (`python -c "from generator import
        OllamaGenerator; print(OllamaGenerator().is_available())"`)."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.ok
        except Exception:
            return False

    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)
        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()
        last_err = None
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
                        "format": "json",       # Ollama enforces valid JSON output
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(cleaned)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]
                return GeneratedAnswer(
                    answer_text=parsed.get("answer", ""),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms, attempts=attempt,
                )
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                last_err = e
                time.sleep(0.2 * attempt)  # simple backoff
                continue
        # all retries exhausted (Ollama not running, model not pulled,
        # malformed output, etc.) -> deterministic fallback rather than
        # error the whole request out
        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb
class ClaudeGenerator:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.fallback = ExtractiveFallbackGenerator()
    def _build_context(self, retrieval: RetrievalResult) -> str:
        blocks = []
        for rc in retrieval.retrieved:
            text = rc.chunk.parent_text or rc.chunk.text
            blocks.append(f"[chunk_id={rc.chunk.chunk_id}] {text}")
        return "\n\n".join(blocks)
    def generate(self, query: str, retrieval: RetrievalResult, max_retries: int = 2) -> GeneratedAnswer:
        if not self.api_key or not retrieval.retrieved:
            return self.fallback.generate(query, retrieval)
        context = self._build_context(retrieval)
        user_msg = f"Context chunks:\n{context}\n\nQuestion: {query}"
        t0 = time.perf_counter()
        last_err = None
        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(
                    ANTHROPIC_URL,
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
                cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(cleaned)
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_cits = parsed.get("citations", [])
                cits = [clean_citation_id(c) for c in raw_cits if clean_citation_id(c)]
                return GeneratedAnswer(
                    answer_text=parsed.get("answer", ""),
                    citations=cits,
                    grounded=not parsed.get("abstain", False),
                    abstained=bool(parsed.get("abstain", False)),
                    latency_ms=latency_ms, attempts=attempt,
                )
            except Exception as e:
                last_err = e
                time.sleep(0.2 * attempt)  # simple backoff
                continue
        # all retries exhausted -> deterministic fallback rather than error out
        fb = self.fallback.generate(query, retrieval)
        fb.attempts = max_retries + 1
        return fb
def make_generator(backend: str | None = None):
    """Factory so the rest of the pipeline (harness/cli/benchmark)
    doesn't need to know which backend is active.
    backend: "ollama" (default, offline/local), "claude", or "extractive"
    (force the deterministic fallback, e.g. for pure-retrieval latency
    testing). Also read from the GENERATOR_BACKEND env var if not passed
    explicitly.
    """
    backend = (backend or os.environ.get("GENERATOR_BACKEND", "ollama")).lower()
    if backend == "ollama":
        gen = OllamaGenerator()
        if not gen.is_available():
            return ExtractiveFallbackGenerator()
        return gen
    if backend == "claude":
        return ClaudeGenerator()
    if backend == "extractive":
        return ExtractiveFallbackGenerator()
    raise ValueError(f"unknown GENERATOR_BACKEND '{backend}' (expected ollama|claude|extractive)")
