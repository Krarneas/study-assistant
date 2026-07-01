"""
config.py — Centralized application configuration.

HARNESS ENGINEERING:
    A harness is the infrastructure that wires everything together reliably.
    Centralizing config here means:
      - One place to change settings (no hunting through 10 files)
      - Secrets stay in .env, never in source code
      - All components read from the same source of truth
      - Easy to swap models or paths without touching business logic
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file so os.getenv() can see it ──────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
# Build all paths relative to THIS file's location so the app works
# regardless of where the user runs it from.
BASE_DIR: Path = Path(__file__).resolve().parent.parent   # project root
UPLOAD_DIR: Path = BASE_DIR / "uploads"
CHROMA_DIR: Path = BASE_DIR / "database" / "chroma"

# Make sure those directories exist at import time
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# The model used for chat completions (answers)
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# The model used to turn text into embedding vectors
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── PDF chunking ──────────────────────────────────────────────────────────────
# CONTEXT ENGINEERING: chunk size controls how much raw text lands in each
# ChromaDB document.  Smaller chunks = more precise retrieval but more of them.
# 500 tokens ≈ ~375 words — a comfortable paragraph-level unit.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))

# Overlap keeps context across chunk boundaries so a sentence that spans
# two chunks is not lost.
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
# How many chunks to pull from ChromaDB on the first retrieval attempt.
TOP_K_INITIAL: int = int(os.getenv("TOP_K_INITIAL", "3"))

# How many extra chunks to pull on each subsequent loop iteration.
TOP_K_EXTRA: int = int(os.getenv("TOP_K_EXTRA", "3"))

# ── Loop Engineering ──────────────────────────────────────────────────────────
# LOOP ENGINEERING: cap iterations so the loop never runs forever.
# Each iteration costs an LLM call, so we keep it small.
MAX_LOOP_ITERATIONS: int = int(os.getenv("MAX_LOOP_ITERATIONS", "3"))

# ── Conversation memory ───────────────────────────────────────────────────────
# CONTEXT ENGINEERING: only keep the last N exchanges in the prompt.
# Sending the full history would blow the context window on long sessions.
MAX_HISTORY_EXCHANGES: int = int(os.getenv("MAX_HISTORY_EXCHANGES", "3"))

# ── LLM generation parameters ─────────────────────────────────────────────────
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

# ── Retry settings (Harness Engineering) ─────────────────────────────────────
# If the OpenAI API returns a transient error we retry rather than crash.
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY_SECONDS", "2.0"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── ChromaDB collection name ──────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "lecture_notes")

# ── FastAPI ───────────────────────────────────────────────────────────────────
FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
