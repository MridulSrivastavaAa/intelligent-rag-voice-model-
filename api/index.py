import os
import sys

# Ensure voice_rag directory is in sys.path when deployed from root
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
voice_rag_dir = os.path.join(root_dir, "voice_rag")
if voice_rag_dir not in sys.path:
    sys.path.insert(0, voice_rag_dir)

from app import app
