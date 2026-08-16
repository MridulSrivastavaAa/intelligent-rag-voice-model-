"""
CLI entry point.
Usage:
  python cli.py --text "मधुमेह के लक्षण क्या हैं?"
  python cli.py --audio path/to/question.wav
"""
from __future__ import annotations
import sys
import argparse
import json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from data_loader import load_sample_docs
from chunking import build_all_chunks
from retrieval import HybridMultiStrategyRetriever
from harness import VoiceRAGHarness
from stt import SarvamSTT
from generator import make_generator
def build_harness(top_k: int = 4) -> VoiceRAGHarness:
    docs = load_sample_docs()
    chunks = build_all_chunks(docs)
    retriever = HybridMultiStrategyRetriever(chunks)
    return VoiceRAGHarness(retriever=retriever, stt=SarvamSTT(),
                            generator=make_generator(), top_k=top_k)
from stt import record_microphone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice-Enabled RAG Pipeline CLI")
    parser.add_argument("--text", type=str, help="question text in quotes")
    parser.add_argument("--audio", type=str, help="path to an audio file (requires SARVAM_API_KEY)")
    parser.add_argument("--listen", "--mic", action="store_true", help="record question from microphone")
    parser.add_argument("--interactive", "-i", action="store_true", help="start an interactive Q&A session")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--duration", type=float, default=5.0, help="microphone recording duration in seconds")
    parser.add_argument("--no-play", action="store_true", help="suppress automatic spoken audio playback")
    args = parser.parse_args()

    harness = build_harness(top_k=args.top_k)

    if args.listen:
        audio_file = record_microphone(duration=args.duration)
        if audio_file:
            resp = harness.run(audio_path=audio_file, play_audio=not args.no_play)
            if resp.transcription and resp.transcription.text:
                print(f"🗣️ Transcribed Voice Question: {resp.transcription.text}")
            if resp.answer:
                print(f"\n💡 Answer: {resp.answer.answer_text}\n")
            print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.interactive or (not args.text and not args.audio):
        print("================================================================")
        print("🎙️ Voice RAG Pipeline - Interactive Mode")
        print("• Type any question in Hindi/English, OR")
        print("• Press ENTER to speak into your microphone, OR")
        print("• Type 'exit' to quit.")
        print("================================================================\n")
        while True:
            try:
                user_input = input("👉 Question [Press ENTER to speak / Type text]: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                if not user_input:
                    # User pressed ENTER -> record from microphone
                    audio_file = record_microphone(duration=args.duration)
                    if not audio_file:
                        continue
                    resp = harness.run(audio_path=audio_file, play_audio=not args.no_play)
                else:
                    resp = harness.run(mock_text=user_input, play_audio=not args.no_play)

                if resp.transcription and resp.transcription.text:
                    print(f"🗣️ Transcribed Query: {resp.transcription.text}")
                if resp.answer:
                    print(f"💡 Answer: {resp.answer.answer_text}\n")
                elif resp.error:
                    print(f"⚠️ Notice: {resp.error}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
        sys.exit(0)

    resp = harness.run(audio_path=args.audio, mock_text=args.text, play_audio=not args.no_play)

    if resp.tts and resp.tts.audio_path and not args.no_play:
        print(f"\n[tts] Playing spoken voice reply from: {resp.tts.audio_path}\n")

    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))




