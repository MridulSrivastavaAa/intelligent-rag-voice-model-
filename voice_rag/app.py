"""
FastAPI Backend Server for Voice-Enabled RAG Pipeline.

Exposes REST endpoints for Web UI:
- POST /api/query: Accepts text or audio questions, executes VoiceRAGHarness, returns response + audio.
- GET /api/documents: Returns indexed MSMARCO-XI dataset passages.
- GET /api/health: Returns system status, active LLM generator, and STT/TTS details.
- GET /api/stats: Returns indexing and strategy statistics.
- Serves static frontend from static/ directory.
"""
from __future__ import annotations
import os
import sys
import base64
import tempfile
import time
from typing import Optional, List, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import load_sample_docs
from chunking import build_all_chunks
from retrieval import HybridMultiStrategyRetriever
from harness import VoiceRAGHarness
from stt import SarvamSTT
from tts import SarvamTTS
from generator import make_generator

# Initialize FastAPI App
app = FastAPI(title="Voice RAG API", version="2.0.0")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline Objects
print("[server] Initializing Voice RAG Pipeline Index & Models...")
DOCS = load_sample_docs()
CHUNKS = build_all_chunks(DOCS)
RETRIEVER = HybridMultiStrategyRetriever(CHUNKS)
STT = SarvamSTT()
TTS = SarvamTTS()
GENERATOR = make_generator()
HARNESS = VoiceRAGHarness(retriever=RETRIEVER, stt=STT, tts=TTS, generator=GENERATOR)


def get_generator_info(gen=None) -> dict:
    active_gen = gen or GENERATOR
    cname = active_gen.__class__.__name__
    name_map = {
        "GroqGenerator": "Groq (Llama 3.3 70B)",
        "GeminiGenerator": "Google Gemini 2.0 Flash",
        "OpenAIGenerator": "OpenAI GPT-4o-mini",
        "ClaudeGenerator": "Anthropic Claude 3.5",
        "OpenRouterGenerator": "OpenRouter",
        "OllamaGenerator": "Ollama (Local LLM)",
        "SmartExtractiveGenerator": "Smart Extractive QA",
        "ExtractiveFallbackGenerator": "Smart Extractive QA",
    }
    return {
        "class": cname,
        "name": name_map.get(cname, cname),
        "is_cloud_llm": cname not in ("SmartExtractiveGenerator", "ExtractiveFallbackGenerator", "OllamaGenerator"),
        "model": getattr(active_gen, "model", "extractive"),
    }


print(f"[server] Pipeline ready! Indexed {len(DOCS)} docs into {len(CHUNKS)} chunks. Active Generator: {get_generator_info()['name']}")


class QueryRequest(BaseModel):
    text: Optional[str] = None
    top_k: int = 4


@app.get("/api/health")
def health_check():
    # Refresh generator if env vars were populated after cold start
    current_gen = make_generator()
    gen_info = get_generator_info(current_gen)
    return {
        "status": "ok",
        "docs_count": len(DOCS),
        "chunks_count": len(CHUNKS),
        "generator_backend": gen_info["name"],
        "generator_class": gen_info["class"],
        "generator_model": gen_info["model"],
        "is_cloud_llm": gen_info["is_cloud_llm"],
        "stt_provider": getattr(STT, 'provider', 'sarvam'),
        "tts_speaker": getattr(TTS, 'speaker', 'anushka'),
        "top_k_default": HARNESS.top_k,
    }


@app.get("/api/stats")
def get_stats():
    strategies = {}
    for c in CHUNKS:
        strat = getattr(c, "strategy", "unknown")
        strategies[strat] = strategies.get(strat, 0) + 1
    gen_info = get_generator_info()
    return {
        "docs_count": len(DOCS),
        "chunks_count": len(CHUNKS),
        "strategies": strategies,
        "generator": gen_info["name"],
        "generator_class": gen_info["class"],
        "stt_provider": getattr(STT, 'provider', 'sarvam'),
        "tts_speaker": getattr(TTS, 'speaker', 'anushka'),
    }


@app.get("/api/documents")
def get_documents() -> List[Dict]:
    return DOCS


@app.post("/api/query")
async def process_query(request: Request):
    temp_audio_path = None
    try:
        content_type = request.headers.get("content-type", "")
        text = None
        top_k = 4
        audio_bytes = None
        audio_filename = None

        if "application/json" in content_type:
            data = await request.json()
            text = data.get("text")
            top_k = int(data.get("top_k", 4))
        else:
            form = await request.form()
            text = form.get("text")
            if form.get("top_k"):
                try:
                    top_k = int(form.get("top_k"))
                except (ValueError, TypeError):
                    top_k = 4
            audio_field = form.get("audio")
            if audio_field and hasattr(audio_field, "read"):
                audio_bytes = await audio_field.read()
                audio_filename = getattr(audio_field, "filename", "audio.wav")

        mock_text = text.strip() if text and str(text).strip() else None

        if audio_bytes:
            suffix = ".wav"
            if audio_filename and os.path.splitext(audio_filename)[1]:
                suffix = os.path.splitext(audio_filename)[1].lower()
            elif "webm" in content_type:
                suffix = ".webm"
            elif "mp3" in content_type:
                suffix = ".mp3"
            elif "ogg" in content_type:
                suffix = ".ogg"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name

        # Ensure latest generator configuration
        HARNESS.generator = make_generator()

        # Execute harness without blocking server playback (play_audio=False)
        resp = HARNESS.run(
            audio_path=temp_audio_path,
            mock_text=mock_text,
            play_audio=False,
            top_k=top_k
        )
        resp_dict = resp.model_dump()

        # Attach active generator metadata
        gen_info = get_generator_info(HARNESS.generator)
        resp_dict["generator_info"] = gen_info

        # Attach Base64 Audio if TTS audio was generated
        if resp.tts and resp.tts.audio_path and os.path.exists(resp.tts.audio_path):
            with open(resp.tts.audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                resp_dict["audio_base64"] = audio_b64

        return resp_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


# Static directory resolution for local and Vercel environments
_STATIC_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "static"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_rag", "static"),
    os.path.join(os.getcwd(), "voice_rag", "static"),
    os.path.join(os.getcwd(), "static"),
]

STATIC_DIR = _STATIC_CANDIDATES[0]
for d in _STATIC_CANDIDATES:
    if os.path.isdir(d):
        STATIC_DIR = d
        break

os.makedirs(STATIC_DIR, exist_ok=True)

# Serve Frontend static files
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)