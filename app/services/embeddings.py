import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from app import config

class EmbeddingError(Exception):
    """Raised when embedddings can't be created. Carries an  HTTP code."""

    def __init__(self, message:str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it"""
    if not config.GEMINI_API_KEY:
        raise EmbeddingError("GEMIN_API_KEY is missing. Check your .env file.", status_code=500)
    return genai.Client(api_key=config.GEMINI_API_KEY)

def _embed_config(task_type: str):
    if config.EMBEDDING_MODEL.endswith("embedding-001"):
        return types.EmbedContentConfig(task_type=task_type)
    return None

def _embed_batch(batch: list[str], task_type: str) -> list[list[float]]:
    """Embed one batch, retrying with backoff on temporary errors."""
    client = get_client()
    last_attempt = config.EMBED_MAX_RETRIES - 1

    for attempt in range(config.EMBED_MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model = config.EMBEDDING_MODEL,
                contents = batch,
                config = _embed_config(task_type),
            )
            vectors = [e.values for e in result.embeddings]
            if len(vectors) != len(batch):
                raise EmbeddingError("Gemini returned the wrong number of embeddings.")
            return vectors
        
        except errors.APIError as e:
            message = e.message or str(e)
            retryable = e.code in (429, 500, 503)
            no_free_quota = "limit: 0" in str(e)

            if not retryable or no_free_quota or attempt == last_attempt:
                status = 429 if e.code == 429 else 502
                raise EmbeddingError(f"Gemini embedding error {e.code}: {message}", status_code=status)

            wait = 5 * 2 ** attempt
            print(f"[embeddings] error {e.code}, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{config.EMBED_MAX_RETRIES})")
            time.sleep(wait)

    raise EmbeddingError("Embedding failed after several retries.")

def embed_texts(texts: list[str], task_type:str) -> list[list[float]]:
    """Embed any number of texts in small batches, pausing between batches."""
    vectors: list[list[float]] = []
    size= config.EMBED_BATCH_SIZE

    for start in range(0, len(texts), size):
        if start > 0:
            time.sleep(config.EMBED_DELAY_SECONDS)
        vectors.extend(_embed_batch(texts[start: start + size], task_type))
    return vectors

def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed documents chunks (used when uploading)"""
    return embed_texts(texts, "RETRIEVAL_DOCUMENT")

def embed_query(text: str) -> list[float]:
    """Embed a user's question (used in Step 5)."""
    return embed_texts([text], "RETRIEVAL_QUERY")[0]


