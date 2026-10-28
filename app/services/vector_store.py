from functools import lru_cache

import chromadb

from app import config
from app.services.chunking import Chunk

@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the local ChromaDB collection stored in chroma_db/."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def replace_document(filename: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Store a document's chunks, replacing any older version of the same file."""
    collection = get_collection()
    collection.delete(where={"source": filename})
    collection.add(
        ids=[f"{filename}::{c.chunk_index}" for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": filename, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ],
    )

def delete_document(filename: str) -> int:
    """Remove all chunks of a file. Returns how many were removed."""
    collection = get_collection()
    ids = collection.get(where={"source": filename}, include=[])["ids"]
    if ids:  # Chroma raises an error if you delete an empty list
        collection.delete(ids=ids)
    return len(ids)

def list_documents() -> list[dict]:
    """Return each stored file with its number of chunks."""
    data = get_collection().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return [{"filename": name, "chunks":n} for name,n in sorted(counts.items())]    


def count() -> int:
    return get_collection().count()

def clear_all() -> int:
    """Remove every chunk from the database. Returns how many were removed."""
    collection = get_collection()
    ids = collection.get(include=[])["ids"]
    if ids:  # Chroma raises an error if you delete an empty list
        collection.delete(ids=ids)
    return len(ids)