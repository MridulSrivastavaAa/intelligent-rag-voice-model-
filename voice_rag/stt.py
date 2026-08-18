"""
Speech-to-text stage: Sarvam AI.

Uses the /speech-to-text REST endpoint (model=saaras:v3, mode=transcribe),
which supports Hindi and 10 other Indian languages plus English
(hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, gu-IN,
en-IN) — a good match for MSMARCO-XI. Auth is via the
`api-subscription-key` header and a multipart file upload.

This sandbox cannot reach api.sarvam.ai (network egress is restricted to
package registries) and no audio input is available in a text chat, so
`transcribe()` falls back to a mock mode that returns the provided text
immediately with a realistic simulated network+inference latency when no
SARVAM_API_KEY/real audio path is supplied. The real HTTP call path is
fully implemented and is what runs when a key + audio file are present.
"""
from __future__ import annotations
import os
import time
import random
import requests
from dotenv import load_dotenv
from schema import TranscriptionResult

load_dotenv()

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT:
    def __init__(self, api_key: str | None = None, model: str = "saaras:v3",
                 language_code: str = "hi-IN", mode: str = "transcribe"):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model
        self.language_code = language_code
        self.mode = mode

    def _mock_latency_ms(self, audio_seconds: float) -> float:
        # Rough real-world figures for short-utterance REST STT: network
        # round trip + inference, independent of our local RAG latency
        # budget. Randomized within a realistic band for benchmarking.
        base = 180 + audio_seconds * 40
        return base + random.uniform(-20, 40)

    def transcribe(self, audio_path: str | None = None, mock_text: str | None = None,
                    audio_seconds: float = 3.0) -> TranscriptionResult:
        t0 = time.perf_counter()

        if audio_path and self.api_key:
            ext = os.path.splitext(audio_path)[1].lower()
            mime_types = {
                ".webm": "audio/webm",
                ".mp3": "audio/mp3",
                ".m4a": "audio/m4a",
                ".ogg": "audio/ogg",
                ".wav": "audio/wav"
            }
            mime_type = mime_types.get(ext, "audio/wav")
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, mime_type)}
                data = {"model": self.model, "mode": self.mode,
                        "language_code": self.language_code}
                headers = {"api-subscription-key": self.api_key}
                resp = requests.post(SARVAM_STT_URL, headers=headers,
                                      files=files, data=data, timeout=30)
                resp.raise_for_status()
                payload = resp.json()

            latency_ms = (time.perf_counter() - t0) * 1000
            return TranscriptionResult(
                text=payload.get("transcript", ""),
                language_code=payload.get("language_code", self.language_code),
                provider="sarvam", latency_ms=latency_ms, is_mocked=False,
            )

        # --- mock path (no key / no audio file / offline sandbox) ---
        simulated_ms = self._mock_latency_ms(audio_seconds)
        time.sleep(min(simulated_ms, 5) / 1000.0)  # tiny sleep so timing is real, capped for test speed
        latency_ms = (time.perf_counter() - t0) * 1000 + max(0, simulated_ms - min(simulated_ms, 5))
        return TranscriptionResult(
            text=mock_text or "", language_code=self.language_code,
            confidence=0.95, provider="sarvam", latency_ms=latency_ms, is_mocked=True,
        )


def record_microphone(duration: float = 5.0, sample_rate: int = 16000,
                      output_path: str = "mic_input.wav") -> str:
    """Record microphone audio using sounddevice and scipy.io.wavfile."""
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        print(f"\n🔴 Recording microphone for {duration:.0f} seconds... Speak your question now!")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        wav.write(output_path, sample_rate, recording)
        print("✅ Recording saved! Transcribing audio with Sarvam AI...\n")
        return output_path
    except Exception as e:
        print(f"⚠️ Microphone recording error: {e}")
        return ""

