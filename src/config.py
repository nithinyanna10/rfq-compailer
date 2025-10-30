import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PROVIDER = os.getenv("PROVIDER", "ollama")  # options: openai, ollama
MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:27b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rfq_docs")
