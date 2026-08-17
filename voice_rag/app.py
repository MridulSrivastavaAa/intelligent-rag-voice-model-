"""
FastAPI Backend Server for Voice-Enabled RAG Pipeline.

Exposes REST endpoints for Web UI:
- POST /api/query: Accepts text or audio questions, executes VoiceRAGHarness, returns response + audio.
- GET /api/documents: Returns indexed MSMARCO-XI dataset passages.
- GET /api/health: Returns system status.
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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
app = FastAPI(title="Voice RAG API", version="1.0.0")

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
print(f"[server] Pipeline ready! Indexed {len(DOCS)} docs into {len(CHUNKS)} chunks. Generator: {GENERATOR.__class__.__name__}")


class QueryRequest(BaseModel):
    text: Optional[str] = None
    top_k: int = 4


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "docs_count": len(DOCS),
        "chunks_count": len(CHUNKS),
        "generator_backend": GENERATOR.__class__.__name__,
        "stt_provider": STT.provider if hasattr(STT, 'provider') else "sarvam",
        "tts_speaker": TTS.speaker if hasattr(TTS, 'speaker') else "anushka",
    }


@app.get("/api/documents")
def get_documents() -> List[Dict]:
    return DOCS


@app.post("/api/query")
async def process_query(
    text: Optional[str] = Form(None),
    top_k: int = Form(4),
    audio: Optional[UploadFile] = File(None)
):
    temp_audio_path = None
    try:
        mock_text = text.strip() if text and text.strip() else None
        
        if audio:
            # Save uploaded browser audio to temporary file
            suffix = ".wav" if audio.content_type and "wav" in audio.content_type else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await audio.read()
                tmp.write(content)
                temp_audio_path = tmp.name

        # Execute harness without blocking server playback (play_audio=False)
        resp = HARNESS.run(audio_path=temp_audio_path, mock_text=mock_text, play_audio=False)
        resp_dict = resp.model_dump()

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


# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve Frontend static files
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
