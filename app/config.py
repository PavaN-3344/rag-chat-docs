import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

def ensure_directories() -> None:
    """Create the local folders if they don't exist yet."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBED_BATCH_SIZE = 20
EMBED_DELAY_SECONDS = 1.0
EMBED_MAX_RETRIES = 5
MAX_CHUNKS_PER_UPLOAD = 500
COLLECTION_NAME = "documents"

TOP_K = 4
GENERATION_TEMPERATURE = 0.2
MAX_QUESTION_CHARS = 1000
GENERATION_MAX_RETRIES = 4