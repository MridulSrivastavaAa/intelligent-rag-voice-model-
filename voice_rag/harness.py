"""
The harness: structured orchestration around the model, as required by
the spec (not a single raw prompt-in/text-out call).
Pipeline stages, each independently timed, retried, and error-handled:
  1. speech-to-text          (SarvamSTT)
  2. input guardrail: unsafe  (InputGuardrail.check_unsafe)  -> may short-circuit
  3. retrieval                (HybridMultiStrategyRetriever)
  4. input guardrail: off-topic (InputGuardrail.check_off_topic) -> may short-circuit
  5. answer generation        (ClaudeGenerator, with internal retry)
  6. output guardrail: groundedness + citation check -> may downgrade to refusal
Every stage wraps its call in `_run_stage`, which:
  - retries transient failures up to `max_retries` with backoff
  - always records a StageTiming (latency, attempts, ok/error)
  - converts unexpected exceptions into a structured error response
    instead of raising out of the pipeline
"""
from __future__ import annotations
import time
import uuid
from typing import Callable, TypeVar
from schema import PipelineResponse, StageTiming, TranscriptionResult
from stt import SarvamSTT
from retrieval import HybridMultiStrategyRetriever
from guardrails import InputGuardrail, OutputGuardrail
from generator import make_generator
T = TypeVar("T")
class StageError(Exception):
    pass
def _run_stage(name: str, fn: Callable[[], T], timings: list[StageTiming],
                max_retries: int = 1) -> T:
    last_exc = None
    for attempt in range(1, max_retries + 2):
        t0 = time.perf_counter()
        try:
            result = fn()
            latency_ms = (time.perf_counter() - t0) * 1000
            timings.append(StageTiming(stage=name, latency_ms=latency_ms, ok=True, attempts=attempt))
            return result
        except Exception as e:  # noqa: BLE001 - convert all stage failures uniformly
            last_exc = e
            latency_ms = (time.perf_counter() - t0) * 1000
            timings.append(StageTiming(stage=name, latency_ms=latency_ms, ok=False,
                                        attempts=attempt, error=str(e)))
    raise StageError(f"stage '{name}' failed after {max_retries + 1} attempts: {last_exc}")
from tts import SarvamTTS


class VoiceRAGHarness:
    def __init__(self, retriever: HybridMultiStrategyRetriever,
                 stt: SarvamSTT | None = None,
                 tts: SarvamTTS | None = None,
                 generator=None,
                 top_k: int = 4):
        self.retriever = retriever
        self.stt = stt or SarvamSTT()
        self.tts = tts or SarvamTTS()
        self.generator = generator or make_generator()
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.top_k = top_k

    def run(self, audio_path: str | None = None, mock_text: str | None = None,
            audio_seconds: float = 3.0, play_audio: bool = True,
            top_k: int | None = None) -> PipelineResponse:
        request_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()
        timings: list[StageTiming] = []
        resp = PipelineResponse(request_id=request_id, query_text=mock_text or "")
        k = top_k if top_k is not None else self.top_k

        try:
            # 1. STT
            transcription: TranscriptionResult = _run_stage(
                "stt",
                lambda: self.stt.transcribe(audio_path=audio_path, mock_text=mock_text,
                                             audio_seconds=audio_seconds),
                timings, max_retries=1,
            )
            resp.transcription = transcription
            resp.query_text = transcription.text

            if not transcription.text.strip():
                resp.status = "refused"
                resp.error = "empty transcription"
                return self._finalize(resp, timings, t_start)

            # 2. input guardrail: unsafe content (fail fast, no retrieval cost spent)
            unsafe_verdict = _run_stage(
                "guardrail_unsafe", lambda: self.input_guard.check_unsafe(transcription.text),
                timings, max_retries=0,
            )
            resp.input_guardrail = unsafe_verdict
            if not unsafe_verdict.passed:
                resp.status = "refused"
                resp.error = "input flagged as unsafe/inappropriate"
                return self._finalize(resp, timings, t_start)

            # 3. retrieval
            retrieval = _run_stage(
                "retrieval", lambda: self.retriever.retrieve(transcription.text, top_k=k),
                timings, max_retries=1,
            )
            resp.retrieval = retrieval

            # 4. input guardrail: off-topic / out-of-domain
            topic_verdict = _run_stage(
                "guardrail_off_topic", lambda: self.input_guard.check_off_topic(retrieval),
                timings, max_retries=0,
            )
            # keep the stricter of the two input verdicts for reporting
            resp.input_guardrail = topic_verdict if not topic_verdict.passed else resp.input_guardrail
            if not topic_verdict.passed:
                resp.status = "refused"
                resp.answer = None
                resp.error = "query judged out-of-domain for this dataset"
                return self._finalize(resp, timings, t_start)

            # 5. generation (has its own internal retry/backoff against the LLM API)
            answer = _run_stage(
                "generation", lambda: self.generator.generate(transcription.text, retrieval),
                timings, max_retries=1,
            )
            resp.answer = answer

            # 6. output guardrails: groundedness + citation validity
            grounded_verdict = _run_stage(
                "guardrail_groundedness",
                lambda: self.output_guard.check_groundedness(answer, retrieval),
                timings, max_retries=0,
            )
            citation_verdict = _run_stage(
                "guardrail_citations",
                lambda: self.output_guard.check_citations(answer, retrieval),
                timings, max_retries=0,
            )
            resp.output_guardrail = grounded_verdict if not grounded_verdict.passed else citation_verdict

            if not grounded_verdict.passed and not answer.abstained:
                # downgrade to a safe refusal rather than surface a possibly
                # hallucinated / ungrounded answer
                resp.answer.answer_text = (
                    "I found related content but couldn't confidently ground a full answer "
                    "in the retrieved passages, so I'm not going to guess."
                )
                resp.answer.abstained = True
                resp.status = "refused"
            elif not citation_verdict.passed and not answer.abstained:
                resp.answer.answer_text = (
                    "I found related content but the response failed citation verification."
                )
                resp.answer.abstained = True
                resp.status = "refused"
            else:
                resp.status = "ok"

            # 7. Text-to-Speech (TTS) Voice Reply Synthesis & Playback
            if resp.status == "ok" and resp.answer and resp.answer.answer_text and not resp.answer.abstained:
                tts_result = _run_stage(
                    "tts",
                    lambda: self.tts.synthesize(resp.answer.answer_text, play_audio=play_audio),
                    timings, max_retries=1,
                )
                resp.tts = tts_result

            return self._finalize(resp, timings, t_start)

        except StageError as e:
            resp.status = "error"
            resp.error = str(e)
            return self._finalize(resp, timings, t_start)

    def _finalize(self, resp: PipelineResponse, timings: list[StageTiming], t_start: float) -> PipelineResponse:
        resp.stage_timings = timings
        resp.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return resp
