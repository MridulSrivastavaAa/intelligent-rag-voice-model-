import os
import sys

# Ensure voice_rag and root directories are in sys.path when deployed on Vercel
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
voice_rag_dir = os.path.join(root_dir, "voice_rag")

for p in [voice_rag_dir, root_dir]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(voice_rag_dir, ".env"))
except ImportError:
    pass

from app import app

