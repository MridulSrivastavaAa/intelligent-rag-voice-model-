"""
Text-to-speech stage: Sarvam AI.

Uses the /text-to-speech REST endpoint (model=bulbul:v2, speaker=anushka),
which supports Hindi and other Indian languages. Auth is via the
`api-subscription-key` header.

Synthesizes generated text answers into spoken WAV audio and plays the audio
through local speakers using `winsound` on Windows.
"""
from __future__ import annotations
import os
import time
import base64
import random
import requests
from dotenv import load_dotenv
from schema import TTSResult

load_dotenv()

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


class SarvamTTS:
    def __init__(self, api_key: str | None = None, model: str = "bulbul:v2",
                 speaker: str = "anushka", target_language_code: str = "hi-IN"):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model
        self.speaker = speaker
        self.target_language_code = target_language_code

    def _play_wav(self, audio_path: str):
        """Plays WAV audio file through local speakers synchronously."""
        if not os.path.exists(audio_path):
            return
        try:
            import winsound
            print(f"[tts] Speaking answer out loud...")
            winsound.PlaySound(audio_path, winsound.SND_FILENAME)
        except Exception as e:
            print(f"[tts] Playback notice: {e}")


    def synthesize(self, text: str, output_path: str = "answer_reply.wav",
                   play_audio: bool = True) -> TTSResult:
        t0 = time.perf_counter()
        clean_text = text.strip()

        if not clean_text:
            return TTSResult(audio_path="", latency_ms=0.0, provider="sarvam",
                             speaker=self.speaker, is_mocked=True)

        if self.api_key:
            try:
                headers = {"api-subscription-key": self.api_key}
                payload = {
                    "inputs": [clean_text],
                    "target_language_code": self.target_language_code,
                    "speaker": self.speaker,
                    "model": self.model,
                }
                resp = requests.post(SARVAM_TTS_URL, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                audios = data.get("audios", [])
                if audios:
                    audio_bytes = base64.b64decode(audios[0])
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)

                    latency_ms = (time.perf_counter() - t0) * 1000

                    if play_audio:
                        self._play_wav(output_path)

                    return TTSResult(
                        audio_path=os.path.abspath(output_path),
                        latency_ms=latency_ms,
                        provider="sarvam",
                        speaker=self.speaker,
                        is_mocked=False
                    )
            except Exception:
                pass  # Fall through to mock/fallback path if network fails

        # --- Fallback / Offline path ---
        simulated_ms = 150.0 + random.uniform(10, 30)
        time.sleep(0.005)
        return TTSResult(
            audio_path="",
            latency_ms=(time.perf_counter() - t0) * 1000 + simulated_ms,
            provider="sarvam",
            speaker=self.speaker,
            is_mocked=True
        )
