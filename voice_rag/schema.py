"""
Structured I/O contracts for the voice-RAG pipeline.

Every stage of the harness consumes and produces one of these typed objects
instead of passing raw strings/dicts around. This is what lets the harness
validate, retry, and log each stage independently.
"""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field
import time


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    strategy: str                     # which chunker produced this chunk
    language: str = "unknown"
    position: int = 0                 # index of chunk within parent doc
    parent_text: Optional[str] = None  # larger surrounding context (parent-doc retrieval)
    metadata: dict = Field(default_factory=dict)


class TranscriptionResult(BaseModel):
    text: str
    language_code: Optional[str] = None
    confidence: Optional[float] = None
    provider: str = "sarvam"
    latency_ms: float = 0.0
    is_mocked: bool = False


class GuardrailVerdict(BaseModel):
    passed: bool
    stage: Literal["input", "output"]
    reasons: list[str] = Field(default_factory=list)
    risk_score: float = 0.0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    source_strategy: str = ""


class RetrievalResult(BaseModel):
    query: str
    retrieved: list[RetrievedChunk]
    latency_ms: float = 0.0
    max_score: float = 0.0
    is_confident: bool = True


class GeneratedAnswer(BaseModel):
    answer_text: str
    citations: list[str] = Field(default_factory=list)   # chunk_ids cited
    grounded: bool = True
    grounding_score: float = 0.0
    abstained: bool = False
    latency_ms: float = 0.0
    attempts: int = 1


class TTSResult(BaseModel):
    audio_path: str = ""
    latency_ms: float = 0.0
    provider: str = "sarvam"
    speaker: str = "anushka"
    is_mocked: bool = False


class StageTiming(BaseModel):
    stage: str
    latency_ms: float
    ok: bool
    attempts: int = 1
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    request_id: str
    query_text: str
    transcription: Optional[TranscriptionResult] = None
    retrieval: Optional[RetrievalResult] = None
    answer: Optional[GeneratedAnswer] = None
    tts: Optional[TTSResult] = None
    input_guardrail: Optional[GuardrailVerdict] = None
    output_guardrail: Optional[GuardrailVerdict] = None
    stage_timings: list[StageTiming] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    status: Literal["ok", "refused", "error"] = "ok"
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

