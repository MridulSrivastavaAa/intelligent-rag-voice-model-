# Voice-Enabled RAG Pipeline (MSMARCO-XI 10,000 Rows) • Google Gemini & Vercel

A complete Voice-Enabled Retrieval-Augmented Generation (RAG) system with hybrid multi-strategy retrieval across **10,000 rows of the MSMARCO-XI dataset**, powered by **Google Gemini API** (Gemini 2.0 / Flash / Pro) and **Sarvam AI Voice STT/TTS**, ready to run locally or deploy to **Vercel**.

## ✨ Key Features
- **Google Gemini API Generation**: Low-latency, grounded answer synthesis with citation tracking and multi-model fallback (`gemini-flash-latest`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-pro-latest`).
- **10,000 MSMARCO-XI Dataset**: Starting 10,000 multilingual rows loaded, indexed, and pre-cached across 31,500+ chunks.
- **Hybrid Multi-Strategy Retrieval**: Dense character n-gram TF-IDF vector space + BM25Okapi lexical matching with Reciprocal Rank Fusion (RRF) and Hinglish query expansion.
- **Multilingual Voice Pipeline**: High-accuracy Speech-to-Text (`saaras:v3`) and natural Hindi/Indian English Text-to-Speech (`bulbul:v2`, Anushka voice) via Sarvam AI.
- **Enterprise Guardrails**: Pre-retrieval safety screening, out-of-domain query detection, and post-generation groundedness & citation verification.
- **Vercel Serverless Ready**: Configured with `@vercel/python` and FastAPI ASGI handler in `api/index.py` with static UI bundling.

## 🚀 Quickstart & Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env file
GEMINI_API_KEY=AQ.Ab8RN6L7dvG_A61eCjaCRsbWTMR5wOIq9DuDlqONvdr22LKqMw
GENERATOR_BACKEND=gemini
GEMINI_MODEL=gemini-flash-latest
SARVAM_API_KEY=sk_bqaf8fve_HqFltnSaJ6ok9jH5uQx4r85e

# 3. Start local development server
uvicorn voice_rag.app:app --reload --port 8000
```
Open your browser at `http://localhost:8000` to interact with the Voice RAG UI.

## ☁️ Deploying to Vercel

1. Push this repository to GitHub.
2. In Vercel, click **Add New Project** and import your repository.
3. In **Settings → Environment Variables**, add:
   - `GEMINI_API_KEY`: `AQ.Ab8RN6L7dvG_A61eCjaCRsbWTMR5wOIq9DuDlqONvdr22LKqMw`
   - `GENERATOR_BACKEND`: `gemini`
   - `GEMINI_MODEL`: `gemini-flash-latest`
   - `SARVAM_API_KEY`: `sk_bqaf8fve_HqFltnSaJ6ok9jH5uQx4r85e`
4. Click **Deploy**. Vercel will build and serve your application live.

## 🧪 Testing

```bash
cd voice_rag
python -m unittest test_pipeline.py
python benchmark.py --n 30
```
