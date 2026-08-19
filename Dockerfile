# ==============================================================================
# Voice RAG Backend Production Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/voice_rag:/app \
    PORT=8000

WORKDIR /app

# Install system dependencies (build-essential, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for efficient Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project source code
COPY . .

# Expose backend API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "voice_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
