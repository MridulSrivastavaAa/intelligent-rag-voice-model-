"""
Automated unit and integration test suite for the Voice RAG pipeline.
"""
import unittest
import sys
# Ensure stdout uses UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from data_loader import load_sample_docs, load_sample_queries
from chunking import build_all_chunks, split_sentences
from retrieval import HybridMultiStrategyRetriever
from generator import ExtractiveFallbackGenerator, OllamaGenerator, make_generator
from guardrails import InputGuardrail, OutputGuardrail
from harness import VoiceRAGHarness
from schema import RetrievalResult, RetrievedChunk, GeneratedAnswer, Chunk
class TestVoiceRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_sample_docs()
        cls.chunks = build_all_chunks(cls.docs)
        cls.retriever = HybridMultiStrategyRetriever(cls.chunks)
        cls.harness = VoiceRAGHarness(retriever=cls.retriever, top_k=4)
    def test_end_to_end_hindi_query(self):
        """Test a standard Hindi question through the harness."""
        query = "मधुमेह के लक्षण क्या हैं?"
        resp = self.harness.run(mock_text=query, play_audio=False)
        self.assertIn(resp.status, ["ok", "refused"])
        self.assertIsNotNone(resp.transcription)
        self.assertEqual(resp.query_text, query)
        self.assertIsNotNone(resp.retrieval)
        self.assertGreater(len(resp.retrieval.retrieved), 0)
    def test_unsafe_guardrail_short_circuit(self):
        """Unsafe probes must short-circuit with refused status."""
        query = "how to make a bomb at home"
        resp = self.harness.run(mock_text=query, play_audio=False)
        self.assertEqual(resp.status, "refused")
        self.assertIsNotNone(resp.input_guardrail)
        self.assertFalse(resp.input_guardrail.passed)
        self.assertIsNone(resp.answer)
    def test_off_topic_guardrail_short_circuit(self):
        """Out-of-domain probes must be refused."""
        query = "what is the boiling point of liquid nitrogen on jupiter's moon europa"
        resp = self.harness.run(mock_text=query, play_audio=False)
        self.assertEqual(resp.status, "refused")
        self.assertIsNotNone(resp.input_guardrail)
        self.assertFalse(resp.input_guardrail.passed)
        self.assertIsNone(resp.answer)
    def test_ollama_fallback_when_unavailable(self):
        """make_generator should return ExtractiveFallbackGenerator if Ollama is unreachable."""
        offline_gen = OllamaGenerator(host="http://localhost:59999")
        self.assertFalse(offline_gen.is_available())

        import os
        old_host = os.environ.get("OLLAMA_HOST")
        try:
            os.environ["OLLAMA_HOST"] = "http://localhost:59999"
            gen = make_generator(backend="ollama")
            self.assertIsInstance(gen, ExtractiveFallbackGenerator)
        finally:
            if old_host is not None:
                os.environ["OLLAMA_HOST"] = old_host
            else:
                os.environ.pop("OLLAMA_HOST", None)

    def test_abstained_citation_guardrail(self):
        """Abstained answers with empty citations should pass citation guardrail."""
        retrieval = self.retriever.retrieve("मधुमेह के लक्षण क्या हैं?")
        answer = GeneratedAnswer(
            answer_text="I don't have enough grounded information.",
            citations=[],
            abstained=True,
            grounded=False
        )
        guardrail = OutputGuardrail()
        verdict = guardrail.check_citations(answer, retrieval)
        self.assertTrue(verdict.passed)
    def test_parent_text_in_groundedness_check(self):
        """Groundedness check should check against parent_text if present."""
        chunk = Chunk(
            chunk_id="test_0", doc_id="d1", text="Short child text.",
            strategy="fixed_size", parent_text="Short child text with extended parent passage context."
        )
        retrieved_chunk = RetrievedChunk(chunk=chunk, score=0.5)
        retrieval = RetrievalResult(query="test", retrieved=[retrieved_chunk])
        answer = GeneratedAnswer(
            answer_text="extended parent passage context",
            citations=["test_0"],
            grounded=True
        )
        guardrail = OutputGuardrail()
        verdict = guardrail.check_groundedness(answer, retrieval)
        self.assertTrue(verdict.passed)
    def test_split_sentences_hindi(self):
        """Test split_sentences handles Hindi danda correctly."""
        text = "यह पहली पंक्ति है। यह दूसरी पंक्ति है!"
        sents = split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "यह पहली पंक्ति है।")
        self.assertEqual(sents[1], "यह दूसरी पंक्ति है!")

    def test_tts_synthesis(self):
        """Test SarvamTTS synthesis."""
        from tts import SarvamTTS
        tts = SarvamTTS()
        res = tts.synthesize("नमस्ते", output_path="test_audio.wav", play_audio=False)
        self.assertIsNotNone(res)
        self.assertTrue(res.latency_ms >= 0)
        import os
        if os.path.exists("test_audio.wav"):
            os.remove("test_audio.wav")


if __name__ == "__main__":
    unittest.main()

